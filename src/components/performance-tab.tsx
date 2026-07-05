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

function pct(value?: number) {
  return `${((value || 0) * 100).toFixed(1)}%`
}

export function PerformanceTab() {
  const [models, setModels] = useState<any[]>([])
  const [datasets, setDatasets] = useState<any>(null)
  const [modelId, setModelId] = useState('')
  const [classifierModelId, setClassifierModelId] = useState('__none__')
  const [recordText, setRecordText] = useState('100,101,102')
  const [maxBeats, setMaxBeats] = useState(1000)
  const [useThresholdOverride, setUseThresholdOverride] = useState(false)
  const [thresholdOverride, setThresholdOverride] = useState(0)
  const [optimizeThreshold, setOptimizeThreshold] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)

  useEffect(() => {
    api.listModels().then(r => {
      const ready = (r.data?.models || []).filter((m: any) => m.status === 'ready' && m.model_path)
      setModels(ready)
      const firstAe = ready.find((m: any) => modelKind(m) === 'autoencoder')
      if (firstAe) {
        setModelId(firstAe.id)
        setThresholdOverride(Number(firstAe.threshold || 0))
      }
    }).catch(console.error)
    api.listDatasets().then(r => setDatasets(r.data)).catch(console.error)
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
  const mitRecords = datasets?.testing?.find((d: any) => d.id === 'mit-bih-arrhythmia')?.records || []

  const runEvaluation = async () => {
    setRunning(true)
    setError(null)
    setResult(null)
    try {
      const records = recordText.split(',').map(r => r.trim()).filter(Boolean)
      const r = await api.evaluateMitbih({
        model_id: modelId,
        classifier_model_id: classifierModelId === '__none__' ? undefined : classifierModelId,
        records,
        max_beats: maxBeats,
        threshold_override: useThresholdOverride ? thresholdOverride : undefined,
        optimize_threshold: optimizeThreshold,
      })
      if (!r.ok) throw new Error(r.detail || 'Evaluation failed')
      setResult(r.data)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setRunning(false)
    }
  }

  const detection = result?.detection
  const confusion = detection?.confusion

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
              Test models on annotated MIT-BIH Arrhythmia records before live analysis.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
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
              <Label>MIT-BIH records</Label>
              <Input
                value={recordText}
                onChange={e => setRecordText(e.target.value)}
                disabled={running}
                className="font-mono text-xs"
              />
              <p className="text-xs text-slate-500">
                Available examples: {mitRecords.slice(0, 8).join(', ')}...
              </p>
            </div>

            <div className="space-y-2">
              <Label>Max annotated beats</Label>
              <Input
                type="number"
                min={1}
                max={10000}
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
                Sweeps thresholds on annotated MIT-BIH beats and selects the best anomaly F1 score.
              </p>
            </div>

            <Button
              onClick={runEvaluation}
              disabled={!modelId || running}
              className="w-full bg-rose-600 hover:bg-rose-700"
            >
              <Play className="w-4 h-4 mr-2" />
              {running ? 'Testing...' : 'Run Performance Test'}
            </Button>

            {error && (
              <Alert variant="destructive">
                <AlertDescription className="text-xs">{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="lg:col-span-3 space-y-4">
        {!result ? (
          <Card className="border-dashed">
            <CardContent className="py-16 text-center text-slate-500">
              <Target className="w-12 h-12 mx-auto mb-4 text-slate-300" />
              <p className="text-lg font-medium">No performance test yet</p>
              <p className="text-sm">Choose models and records, then run a MIT-BIH annotation test.</p>
            </CardContent>
          </Card>
        ) : (
          <>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="w-5 h-5 text-rose-500" />
                  Anomaly Detection Results
                </CardTitle>
                <CardDescription>
                  Binary detection where abnormal MIT-BIH annotation classes count as anomalies.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  <Metric label="Sensitivity" value={pct(detection?.sensitivity)} hint="TP / (TP + FN)" />
                  <Metric label="Specificity" value={pct(detection?.specificity)} hint="TN / (TN + FP)" />
                  <Metric label="Precision" value={pct(detection?.precision)} hint="TP / (TP + FP)" />
                  <Metric label="F1" value={pct(detection?.f1)} hint="Balanced detection score" />
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
                    Best threshold is selected by anomaly F1. Use it in Live Analysis as a custom threshold.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
                    <Metric
                      label="Default threshold"
                      value={result.threshold_optimization.default.threshold.toFixed(4)}
                      hint={`F1 ${pct(result.threshold_optimization.default.f1)}`}
                    />
                    <Metric
                      label="Best threshold"
                      value={result.threshold_optimization.best.threshold.toFixed(4)}
                      hint={`F1 ${pct(result.threshold_optimization.best.f1)}`}
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
                        <TableHead>F1</TableHead>
                        <TableHead>Sensitivity</TableHead>
                        <TableHead>Specificity</TableHead>
                        <TableHead>Precision</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {(result.threshold_optimization.candidates || [])
                        .slice()
                        .sort((a: any, b: any) => b.f1 - a.f1)
                        .slice(0, 10)
                        .map((row: any) => (
                          <TableRow key={row.threshold}>
                            <TableCell className="font-mono">{row.threshold.toFixed(4)}</TableCell>
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
        )}
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
