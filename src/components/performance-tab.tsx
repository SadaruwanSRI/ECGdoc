'use client'

import { useEffect, useMemo, useState } from 'react'
import { api } from '@/lib/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Progress } from '@/components/ui/progress'
import { Activity, BarChart3, Database, Play, Target, Zap } from 'lucide-react'

function modelKind(model: any): 'classifier' | 'autoencoder' {
  try {
    const cfg = JSON.parse(model.config_json || '{}')
    if (cfg.type === 'classifier') return 'classifier'
  } catch {}
  return String(model.architecture || '').toLowerCase().includes('classifier')
    ? 'classifier'
    : 'autoencoder'
}

function isHierarchicalSystem(model: any) {
  try {
    const cfg = JSON.parse(model.config_json || '{}')
    return cfg.system_kind === 'final_mlii_temporal_holdout'
  } catch {
    return false
  }
}

function pct(value?: number | null) {
  return value == null ? '-' : `${(value * 100).toFixed(1)}%`
}

function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

export function PerformanceTab() {
  const [models, setModels] = useState<any[]>([])
  const [datasets, setDatasets] = useState<any>(null)
  const [evaluationDatasets, setEvaluationDatasets] = useState<any[]>([])
  const [finalStudy, setFinalStudy] = useState<any>(null)
  const [datasetId, setDatasetId] = useState('mit-bih-arrhythmia')
  const [leadName, setLeadName] = useState('MLII')
  const [mode, setMode] = useState('offline')
  const [modelId, setModelId] = useState('')
  const [classifierModelId, setClassifierModelId] = useState('__none__')
  const [recordText, setRecordText] = useState('100,101,103')
  const [maxBeats, setMaxBeats] = useState(1000)
  const [useThresholdOverride, setUseThresholdOverride] = useState(false)
  const [thresholdOverride, setThresholdOverride] = useState(0)
  const [optimizeThreshold, setOptimizeThreshold] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)
  const [progress, setProgress] = useState<any>(null)

  useEffect(() => {
    api.listModels().then(r => {
      const ready = (r.data?.models || []).filter((m: any) => m.status === 'ready' && m.model_path)
      setModels(ready)
      const firstAe = ready.find((m: any) => m.id === 'nsrdb-primary-autoencoder') || ready.find((m: any) => modelKind(m) === 'autoencoder')
      if (firstAe) {
        setModelId(firstAe.id)
        setThresholdOverride(Number(firstAe.threshold || 0))
      }
      const hierarchy = ready.find((m: any) => m.id === 'final-mlii-corrected-20260916') || ready.find((m: any) => isHierarchicalSystem(m))
      if (hierarchy) setClassifierModelId(hierarchy.id)
    }).catch(console.error)
    api.listDatasets().then(r => setDatasets(r.data)).catch(console.error)
    api.listEvaluationDatasets().then(r => {
      const items = r.data?.datasets || []
      setEvaluationDatasets(items)
      const first = items.find((d: any) => d.id === 'mit-bih-arrhythmia') || items[0]
      if (first) {
        setDatasetId(first.id)
        setLeadName(first.default_lead)
        setRecordText((first.default_records || []).join(','))
      }
    }).catch(console.error)
    api.getFinalStudy().then(r => setFinalStudy(r.data)).catch(() => {})
  }, [])

  useEffect(() => {
    const selected = models.find(m => m.id === modelId)
    if (selected?.threshold != null) {
      setThresholdOverride(Number(selected.threshold))
    }
  }, [modelId, models])

  const autoencoderModels = useMemo(
    () => models.filter(m => modelKind(m) === 'autoencoder'),
    [models],
  )
  const classifierModels = useMemo(
    () => models.filter(m => modelKind(m) === 'classifier'),
    [models],
  )
  const selectedModel = models.find(m => m.id === modelId)
  const selectedDataset = evaluationDatasets.find(d => d.id === datasetId)
  const testDatasetFallback = datasets?.testing?.find((d: any) => d.id === datasetId)
  const availableRecords = selectedDataset?.records || testDatasetFallback?.records || []
  const allRecords = availableRecords.join(',')

  useEffect(() => {
    if (!selectedDataset) return
    setLeadName(selectedDataset.default_lead || 'MLII')
    setRecordText((selectedDataset.default_records || selectedDataset.records?.slice(0, 3) || []).join(','))
    setMaxBeats(selectedDataset.id === 'incartdb' ? 5000 : 1000)
    setOptimizeThreshold(false)
  }, [selectedDataset])

  const runEvaluation = async () => {
    setRunning(true)
    setError(null)
    setResult(null)
    setProgress(null)
    try {
      const records = recordText.split(',').map(r => r.trim()).filter(Boolean)
      const config = {
        dataset_id: datasetId,
        model_id: modelId,
        classifier_model_id: classifierModelId === '__none__' ? undefined : classifierModelId,
        records,
        max_beats: maxBeats,
        lead_name: leadName,
        mode,
        threshold_override: useThresholdOverride ? thresholdOverride : undefined,
        optimize_threshold: optimizeThreshold,
      }
      const started = await api.startEvaluation(config)
      if (!started.ok || !started.data?.job_id) {
        throw new Error(started.detail || 'Could not start evaluation')
      }

      while (true) {
        const snapshot = await api.getEvaluationProgress(started.data.job_id)
        if (!snapshot.ok) throw new Error(snapshot.detail || 'Could not read evaluation progress')
        setProgress(snapshot.data)

        if (snapshot.data?.status === 'completed') {
          setResult(snapshot.data.result)
          break
        }
        if (snapshot.data?.status === 'failed') {
          throw new Error(snapshot.data.error || 'Evaluation failed')
        }
        await delay(900)
      }
    } catch (e: any) {
      setError(e.message)
    } finally {
      setRunning(false)
    }
  }

  const detection = result?.detection
  const confusion = detection?.confusion
  const progressValue = Math.max(0, Math.min(100, Number(progress?.progress || 0)))
  const liveDetection = progress?.current_detection

  return (
    <div className="grid lg:grid-cols-4 gap-6">
      <div className="lg:col-span-1 space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-rose-500" />
              Performance Test
            </CardTitle>
            <CardDescription>
              Replay annotated beats from an internal or external dataset through the same hierarchy used by live analysis.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Test dataset</Label>
              <Select value={datasetId} onValueChange={setDatasetId} disabled={running}>
                <SelectTrigger><SelectValue placeholder="Select test dataset" /></SelectTrigger>
                <SelectContent>
                  {evaluationDatasets.map(d => (
                    <SelectItem key={d.id} value={d.id}>
                      <div className="flex flex-col">
                        <span>{d.name}</span>
                        <span className="text-xs text-slate-500">{d.validation_kind} | {d.record_count} records | lead {d.default_lead}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedDataset && (
                <p className="text-xs text-slate-500">
                  {selectedDataset.description}
                </p>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label>Lead</Label>
                <Select value={leadName} onValueChange={setLeadName} disabled={running}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {(selectedDataset?.supported_leads || [leadName]).map((lead: string) => (
                      <SelectItem key={lead} value={lead}>{lead}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Mode</Label>
                <Select value={mode} onValueChange={setMode} disabled={running}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="offline">Offline beat-level</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {selectedDataset?.warning && (
              <Alert className={selectedDataset.validation_kind === 'external' ? 'border-amber-200 bg-amber-50' : ''}>
                <AlertDescription className="text-xs">
                  {selectedDataset.warning}
                </AlertDescription>
              </Alert>
            )}

            <div className="space-y-2">
              <Label>Anomaly model</Label>
              <Select value={modelId} onValueChange={setModelId} disabled={running}>
                <SelectTrigger><SelectValue placeholder="Select anomaly model" /></SelectTrigger>
                <SelectContent>
                  {autoencoderModels.map(m => (
                    <SelectItem key={m.id} value={m.id}>
                      <div className="flex flex-col">
                        <span>{m.name}</span>
                        <span className="text-xs text-slate-500 font-mono">{m.version} threshold={m.threshold?.toFixed(4)}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Classifier model</Label>
              <Select value={classifierModelId} onValueChange={setClassifierModelId} disabled={running}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">None, anomaly only</SelectItem>
                  {classifierModels.map(m => (
                    <SelectItem key={m.id} value={m.id}>
                      <span>{m.name} ({m.version})</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Records</Label>
              <Input
                value={recordText}
                onChange={e => setRecordText(e.target.value)}
                disabled={running}
                className="font-mono text-xs"
              />
              <p className="text-xs text-slate-500">
                Available examples: {availableRecords.slice(0, 8).join(', ')}...
              </p>
              <Button
                type="button"
                size="sm"
                variant="outline"
                className="w-full"
                disabled={running || !availableRecords.length}
                onClick={() => {
                  setRecordText(allRecords)
                  setMaxBeats(datasetId === 'incartdb' ? 180000 : 120000)
                }}
              >
                Use all {availableRecords.length || selectedDataset?.record_count || 0} records
              </Button>
            </div>

            <div className="space-y-2">
              <Label>Max annotated beats</Label>
              <Input
                type="number"
                min={1}
                max={120000}
                value={maxBeats}
                onChange={e => setMaxBeats(Number(e.target.value))}
                disabled={running}
              />
            </div>

            <div className="space-y-2 rounded-md border bg-slate-50 p-3">
              <div className="flex items-center justify-between">
                <Label>Threshold</Label>
                <Button
                  size="sm"
                  variant={useThresholdOverride ? 'default' : 'outline'}
                  onClick={() => setUseThresholdOverride(v => !v)}
                  disabled={running}
                  className="h-7 text-xs"
                >
                  {useThresholdOverride ? 'Custom' : 'Model default'}
                </Button>
              </div>
              <Input
                type="number"
                step="0.001"
                min="0"
                value={thresholdOverride}
                onChange={e => setThresholdOverride(Number(e.target.value))}
                disabled={running || !useThresholdOverride}
                className="font-mono text-xs"
              />
              <div className="text-xs text-slate-500">
                Saved: {selectedModel?.threshold?.toFixed(4) || '-'} | lower is more sensitive.
              </div>
            </div>

            <div className="space-y-2 rounded-md border bg-amber-50 p-3">
              <div className="flex items-center justify-between gap-2">
                <Label>Find best threshold</Label>
                <Button
                  size="sm"
                  variant={optimizeThreshold ? 'default' : 'outline'}
                  onClick={() => setOptimizeThreshold(v => !v)}
                  disabled={running}
                  className="h-7 text-xs"
                >
                  {optimizeThreshold ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <p className="text-xs text-slate-600">
                Exploratory: selects balanced accuracy on these same labels. It is not an independent test.
              </p>
            </div>

            <Button
              onClick={runEvaluation}
              disabled={!modelId || running}
              className="w-full bg-rose-600 hover:bg-rose-700"
            >
              <Play className="w-4 h-4 mr-2" />
              {running ? `Testing... ${progressValue.toFixed(0)}%` : 'Run Performance Test'}
            </Button>

            {(running || progress) && (
              <div className="space-y-3 rounded-md border bg-slate-50 p-3">
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="font-medium text-slate-800">Progress</span>
                  <span className="font-mono text-slate-600">{progressValue.toFixed(0)}%</span>
                </div>
                <Progress value={progressValue} />
                <div className="flex items-center justify-between gap-3 text-xs text-slate-500">
                  <span>{progress?.message || 'Starting evaluation'}</span>
                  <span className="font-mono">
                    {Number(progress?.processed || 0).toLocaleString()} / {Number(progress?.target || maxBeats).toLocaleString()}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Metric label="Current accuracy" value={pct(liveDetection?.accuracy)} />
                  <Metric label="Current balanced" value={pct(liveDetection?.balanced_accuracy)} />
                </div>
              </div>
            )}

            {error && (
              <Alert variant="destructive">
                <AlertDescription className="text-xs">{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="lg:col-span-3 space-y-4">
        {finalStudy && (
          <Card className="border-emerald-200 bg-emerald-50/40">
            <CardHeader>
              <CardTitle>Final MLII temporal-holdout benchmark</CardTitle>
              <CardDescription>
                The last 20% of accepted beats tests later ECG from known patients. The corrected rerun uses local window preprocessing.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
                <Metric
                  label="Balanced accuracy"
                  value={pct(finalStudy.test_binary?.balanced_accuracy)}
                  hint="Fixed temporal test"
                />
                <Metric
                  label="AUROC"
                  value={finalStudy.test_binary?.auroc?.toFixed(4) || '-'}
                  hint="Held-out final 20%"
                />
                <Metric
                  label="Binary beats"
                  value={finalStudy.test_binary?.support?.toLocaleString() || '-'}
                  hint={`${finalStudy.test_binary?.abnormal_support?.toLocaleString() || '-'} abnormal`}
                />
                <Metric
                  label="Whole-system macro-F1"
                  value={finalStudy.test_whole_system?.macro_f1?.toFixed(4) || '-'}
                  hint="Whole-system six-class result"
                />
              </div>
              <Alert className="mt-4 border-emerald-200">
                <AlertDescription className="text-xs">
                  This retrospective test was observed during earlier development. Its scores may be optimistic and do not establish performance for new patients.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        )}

        {running && progress && (
          <Card className="border-rose-200 bg-rose-50/40">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-rose-500" />
                Live Performance
              </CardTitle>
              <CardDescription>
                Current metrics from beats already scored in this test run.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-slate-600">{progress.message || 'Running evaluation'}</span>
                  <span className="font-mono text-slate-700">{progressValue.toFixed(0)}%</span>
                </div>
                <Progress value={progressValue} />
              </div>
              <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
                <Metric label="Current accuracy" value={pct(liveDetection?.accuracy)} />
                <Metric label="Balanced accuracy" value={pct(liveDetection?.balanced_accuracy)} />
                <Metric label="Sensitivity" value={pct(liveDetection?.sensitivity)} />
                <Metric label="Specificity" value={pct(liveDetection?.specificity)} />
              </div>
              <div className="grid sm:grid-cols-4 gap-3">
                <Metric label="TP" value={liveDetection?.confusion?.tp ?? 0} />
                <Metric label="TN" value={liveDetection?.confusion?.tn ?? 0} />
                <Metric label="FP" value={liveDetection?.confusion?.fp ?? 0} />
                <Metric label="FN" value={liveDetection?.confusion?.fn ?? 0} />
              </div>
            </CardContent>
          </Card>
        )}

        {!result && !running ? (
          <Card className="border-dashed">
            <CardContent className="py-16 text-center text-slate-500">
              <Target className="w-12 h-12 mx-auto mb-4 text-slate-300" />
              <p className="text-lg font-medium">No performance test yet</p>
              <p className="text-sm">Choose a dataset, model, and records, then run an annotation-based test.</p>
            </CardContent>
          </Card>
        ) : result ? (
          <>
            <Card className={result.dataset?.validation_kind === 'external' ? 'border-amber-200 bg-amber-50/40' : ''}>
              <CardHeader>
                <CardTitle>{result.dataset?.name || 'Evaluation dataset'}</CardTitle>
                <CardDescription>
                  {result.dataset?.validation_kind === 'external' ? 'External unseen validation' : 'Internal-source validation'} | lead {result.dataset?.lead_name} | offline beat-level evaluation
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  <Metric label="Records requested" value={result.records?.filter((r: any) => r.status === 'ok').length || 0} />
                  <Metric label="Annotated beats" value={result.n_beats?.toLocaleString() || 0} />
                  <Metric label="Source sampling" value={`${result.dataset?.sampling_rate_hz || '-'} Hz`} />
                  <Metric label="Lead" value={result.dataset?.lead_name || '-'} />
                </div>
                {result.dataset?.warning && (
                  <Alert className="mt-4 border-amber-200">
                    <AlertDescription className="text-xs">
                      {result.dataset.warning}
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="w-5 h-5 text-rose-500" />
                  Anomaly Detection Results
                </CardTitle>
                <CardDescription>
                  Binary detection where supported ECG beat/rhythm annotation classes count as anomalies.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  <Metric label="Balanced accuracy" value={pct(detection?.balanced_accuracy)} hint="Mean of sensitivity and specificity" />
                  <Metric label="Sensitivity" value={pct(detection?.sensitivity)} hint="TP / (TP + FN)" />
                  <Metric label="Specificity" value={pct(detection?.specificity)} hint="TN / (TN + FP)" />
                  <Metric label="F1" value={pct(detection?.f1)} hint="Precision-recall balance" />
                </div>
                <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 mt-4">
                  <Metric label="AUROC" value={detection?.auroc == null ? '-' : detection.auroc.toFixed(4)} hint="Ranking over every threshold" />
                  <Metric label="AUPRC" value={detection?.auprc == null ? '-' : detection.auprc.toFixed(4)} hint="Useful with class imbalance" />
                  <Metric label="Brier score" value={detection?.brier == null ? '-' : detection.brier.toFixed(4)} hint="Lower probability error is better" />
                  <Metric label="Precision" value={pct(detection?.precision)} hint="TP / (TP + FP)" />
                </div>
                <div className="grid sm:grid-cols-4 gap-3 mt-4">
                  <Metric label="TP" value={confusion?.tp ?? 0} />
                  <Metric label="TN" value={confusion?.tn ?? 0} />
                  <Metric label="FP" value={confusion?.fp ?? 0} />
                  <Metric label="FN" value={confusion?.fn ?? 0} />
                </div>
                <div className="mt-4 flex gap-2 flex-wrap text-sm">
                  <Badge variant="outline">Threshold {result.threshold.toFixed(4)}</Badge>
                  {result.threshold_optimization?.enabled && (
                    <Badge variant="default">Optimized threshold selected</Badge>
                  )}
                  <Badge variant="outline">{result.n_beats} annotated beats</Badge>
                  {Object.entries(result.class_support || {}).map(([cls, count]) => (
                    <Badge key={cls} variant="secondary">{cls}: {String(count)}</Badge>
                  ))}
                </div>
              </CardContent>
            </Card>

            {result.threshold_optimization && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="w-5 h-5 text-amber-500" />
                    Threshold Search
                  </CardTitle>
                  <CardDescription>
                    The displayed search maximizes balanced accuracy on this replay. Treat it as exploratory, not as clinical validation.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
                    <Metric
                      label="Default threshold"
                      value={result.threshold_optimization.default.threshold.toFixed(4)}
                      hint={`Balanced accuracy ${pct(result.threshold_optimization.default.balanced_accuracy)}`}
                    />
                    <Metric
                      label="Best threshold"
                      value={result.threshold_optimization.best.threshold.toFixed(4)}
                      hint={`Balanced accuracy ${pct(result.threshold_optimization.best.balanced_accuracy)}`}
                    />
                    <Metric
                      label="Best sensitivity"
                      value={pct(result.threshold_optimization.best.sensitivity)}
                    />
                    <Metric
                      label="Best specificity"
                      value={pct(result.threshold_optimization.best.specificity)}
                    />
                  </div>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Threshold</TableHead>
                        <TableHead>Balanced acc.</TableHead>
                        <TableHead>F1</TableHead>
                        <TableHead>Sensitivity</TableHead>
                        <TableHead>Specificity</TableHead>
                        <TableHead>Precision</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {(result.threshold_optimization.candidates || [])
                        .slice()
                        .sort((a: any, b: any) => (b.balanced_accuracy ?? -1) - (a.balanced_accuracy ?? -1))
                        .slice(0, 10)
                        .map((row: any) => (
                          <TableRow key={row.threshold}>
                            <TableCell className="font-mono">{row.threshold.toFixed(4)}</TableCell>
                            <TableCell>{pct(row.balanced_accuracy)}</TableCell>
                            <TableCell>{pct(row.f1)}</TableCell>
                            <TableCell>{pct(row.sensitivity)}</TableCell>
                            <TableCell>{pct(row.specificity)}</TableCell>
                            <TableCell>{pct(row.precision)}</TableCell>
                          </TableRow>
                        ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}

            {!!result.abnormal_by_symbol?.length && (
              <Card>
                <CardHeader>
                  <CardTitle>All abnormal annotation types</CardTitle>
                  <CardDescription>
                    Every annotated abnormal beat is included in binary detection, even when it has no supported six-class name.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Annotation symbol</TableHead>
                        <TableHead>Support</TableHead>
                        <TableHead>Detected</TableHead>
                        <TableHead>Missed</TableHead>
                        <TableHead>Recall</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {result.abnormal_by_symbol.map((row: any) => (
                        <TableRow key={row.symbol}>
                          <TableCell className="font-mono">{row.symbol}</TableCell>
                          <TableCell>{row.support}</TableCell>
                          <TableCell>{row.detected}</TableCell>
                          <TableCell>{row.missed}</TableCell>
                          <TableCell>{pct(row.recall)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}

            {!!result.per_record_detection?.length && (
              <Card>
                <CardHeader>
                  <CardTitle>Record-level summary</CardTitle>
                  <CardDescription>
                    Differences between records reveal subject-to-subject variation hidden by one pooled number.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Record</TableHead>
                        <TableHead>Beats</TableHead>
                        <TableHead>Balanced acc.</TableHead>
                        <TableHead>Sensitivity</TableHead>
                        <TableHead>Specificity</TableHead>
                        <TableHead>AUROC</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {result.per_record_detection.map((row: any) => (
                        <TableRow key={row.record}>
                          <TableCell className="font-mono">{row.record}</TableCell>
                          <TableCell>
                            {Object.values(row.confusion || {}).reduce(
                              (total: number, value: any) => total + Number(value),
                              0,
                            )}
                          </TableCell>
                          <TableCell>{row.balanced_accuracy == null ? '-' : pct(row.balanced_accuracy)}</TableCell>
                          <TableCell>{pct(row.sensitivity)}</TableCell>
                          <TableCell>{pct(row.specificity)}</TableCell>
                          <TableCell>{row.auroc == null ? '-' : row.auroc.toFixed(3)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}

            {result.classifier && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="w-5 h-5 text-amber-500" />
                    Classification Results
                  </CardTitle>
                  <CardDescription>
                    Beat class prediction performance from the selected classifier model.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
                    <Metric label="Classifier accuracy" value={pct(result.classifier.accuracy)} />
                    <Metric label="Classifier macro-F1" value={pct(result.classifier.macro_f1)} />
                  </div>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Class</TableHead>
                        <TableHead>Precision</TableHead>
                        <TableHead>Recall</TableHead>
                        <TableHead>F1</TableHead>
                        <TableHead>Support</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {Object.entries(result.classifier.per_class || {}).map(([cls, row]: [string, any]) => (
                        <TableRow key={cls}>
                          <TableCell className="font-mono">{cls}</TableCell>
                          <TableCell>{pct(row.precision)}</TableCell>
                          <TableCell>{pct(row.recall)}</TableCell>
                          <TableCell>{pct(row.f1)}</TableCell>
                          <TableCell>{row.support}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="w-5 h-5 text-slate-500" />
                  Example Predictions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Record</TableHead>
                      <TableHead>True</TableHead>
                      <TableHead>Score</TableHead>
                      <TableHead>Anomaly</TableHead>
                      <TableHead>Class</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(result.examples || []).slice(0, 20).map((ex: any, i: number) => (
                      <TableRow key={`${ex.record}-${ex.sample}-${i}`}>
                        <TableCell>{ex.record}</TableCell>
                        <TableCell>
                          <Badge variant={ex.true_anomaly ? 'destructive' : 'default'}>{ex.true_class}</Badge>
                        </TableCell>
                        <TableCell className="font-mono">{ex.score.toFixed(4)}</TableCell>
                        <TableCell>{ex.predicted_anomaly ? 'Yes' : 'No'}</TableCell>
                        <TableCell>{ex.predicted_class || '-'}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </>
        ) : null}
      </div>
    </div>
  )
}

function Metric({ label, value, hint }: { label: string; value: any; hint?: string }) {
  return (
    <div className="rounded-lg border bg-slate-50 p-3">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="font-mono text-xl font-semibold text-slate-900">{value}</div>
      {hint && <div className="text-xs text-slate-400">{hint}</div>}
    </div>
  )
}
