'use client'

import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Play, Square, Activity, Cpu, Database, Layers, Zap, TrendingDown, CheckCircle2, XCircle, Upload } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'

type Dataset = {
  training: { id: string; name: string; description: string; records: string[] }[]
  testing: { id: string; name: string; description: string; records: string[] }[]
}

type TrainingEvent =
  | { type: 'init'; n_train: number; n_val: number; architecture: any; device: string }
  | { type: 'qc'; dataset: string; n_in?: number; n_kept: number; retention_pct?: number; distribution?: Record<string, number>; qc_method: string; min_quality: number }
  | { type: 'batch'; epoch: number; total_epochs: number; batch: number; total_batches: number; batch_loss: number; elapsed_s: number }
  | { type: 'epoch'; epoch: number; total_epochs: number; train_loss: number; val_loss: number; elapsed_s: number }
  | { type: 'stopped'; epoch: number }
  | { type: 'done'; status: string; final_result?: any; error?: string }

export function TrainingTab() {
  const [datasets, setDatasets] = useState<Dataset | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [runId, setRunId] = useState<string | null>(null)
  const [modelId, setModelId] = useState<string | null>(null)
  const [events, setEvents] = useState<TrainingEvent[]>([])
  const [status, setStatus] = useState<string>('idle')
  // (WebSocket removed — now using direct HTTP polling for training progress)
  const [error, setError] = useState<string | null>(null)
  const [recentRuns, setRecentRuns] = useState<any[]>([])
  const [architecturePreview, setArchitecturePreview] = useState<any | null>(null)

  // Config form state
  const [cfg, setCfg] = useState({
    model_name: `ecg-ae-${new Date().toISOString().slice(0, 10)}`,
    description: 'Trained from web UI',
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.001,
    dataset_name: 'synthetic',
    uploaded_dataset_id: '' as string,
    max_records: 4,
    threshold_k: 2.0,
    use_ecg_qc: true,
    min_quality: 2,
  })

  // Upload state
  const [uploadName, setUploadName] = useState('')
  const [uploadFiles, setUploadFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState<string | null>(null)

  // Models list (for classifier encoder selection)
  const [models, setModels] = useState<any[]>([])

  // Classifier training state
  const [classifierCfg, setClassifierCfg] = useState({
    encoder_model_id: '',
    classifier_name: 'ecg-classifier-v1',
    epochs: 30,
    batch_size: 64,
    learning_rate: 0.001,
    max_records: 48,
    hidden_dim: 256,
    dropout: 0.35,
    freeze_encoder: false,
    encoder_learning_rate: 0.0001,
    focal_gamma: 2.0,
    augment: true,
  })
  const [classifierStatus, setClassifierStatus] = useState<string>('idle')
  const [classifierRunId, setClassifierRunId] = useState<string | null>(null)
  const [classifierModelId, setClassifierModelId] = useState<string | null>(null)
  const [classifierEvents, setClassifierEvents] = useState<any[]>([])
  const [classifierError, setClassifierError] = useState<string | null>(null)
  const [classifierResult, setClassifierResult] = useState<any>(null)

  useEffect(() => {
    api.listDatasets().then(r => setDatasets(r.data)).catch(console.error)
    api.listModels().then(r => setModels(r.data?.models || [])).catch(console.error)
    api.getTrainingArchitectures().then(r => setArchitecturePreview(r.data)).catch(console.error)
    refreshRuns()
  }, [])

  const refreshRuns = async () => {
    try {
      const r = await api.listTrainingRuns()
      const runs = r.data?.runs || []
      setRecentRuns(runs)

      // If there's a running/queued training run and we're not already polling,
      // resume polling so the UI shows real-time progress after a page refresh.
      if (!runId) {
        const activeRun = runs.find((r: any) => r.status === 'running' || r.status === 'queued')
        if (activeRun) {
          if (activeRun.run_type === 'classifier') {
            setClassifierRunId(activeRun.id)
            setClassifierModelId(activeRun.model_id)
            setClassifierStatus('running')
          } else {
            setRunId(activeRun.id)
            setModelId(activeRun.model_id)
            setStatus('running')
          }
        }
      }
    } catch (e) {}
  }

  // Poll the backend directly for training progress (more reliable than WebSocket)
  useEffect(() => {
    if (!runId) return

    let lastEventCount = 0
    let active = true

    const poll = async () => {
      if (!active) return
      try {
        const r = await api.getTrainingRun(runId)
        if (!active || !r.ok || !r.data) return

        const data = r.data
        const progress = data.progress || data.logs || []

        // Only update if there are new events
        if (progress.length > lastEventCount) {
          setEvents(progress.slice(-200))  // keep last 200 events
          lastEventCount = progress.length
        }

        // Update status
        if (data.status === 'completed' || data.status === 'failed' || data.status === 'stopped') {
          setStatus(data.status)
          if (data.status === 'failed') {
            setError(data.error || 'Training failed')
          }
          if (data.final_result) {
            // Training is done — stop polling
            refreshRuns()
            return
          }
        } else if (data.status === 'running' || progress.length > 0) {
          setStatus('running')
        }
      } catch (e) {
        // Ignore polling errors (e.g. transient network issues)
      }

      // Schedule next poll
      if (active) {
        setTimeout(poll, 1000)
      }
    }

    // Start polling immediately
    poll()

    return () => {
      active = false
    }
  }, [runId])

  useEffect(() => {
    if (!classifierRunId) return

    let lastEventCount = 0
    let active = true

    const poll = async () => {
      if (!active) return
      try {
        const r = await api.getTrainingRun(classifierRunId)
        if (!active || !r.ok || !r.data) return

        const data = r.data
        const progress = data.progress || data.logs || []
        if (progress.length > lastEventCount) {
          setClassifierEvents(progress.slice(-200))
          lastEventCount = progress.length
        }

        if (data.status === 'completed' || data.status === 'failed' || data.status === 'stopped') {
          setClassifierStatus(data.status)
          if (data.status === 'failed') {
            setClassifierError(data.error || 'Training failed')
          }
          if (data.final_result) {
            setClassifierResult(data.final_result)
          }
          api.listModels().then(r => setModels(r.data?.models || [])).catch(console.error)
          refreshRuns()
          return
        }

        if (data.status === 'running' || data.status === 'queued' || progress.length > 0) {
          setClassifierStatus(data.status === 'queued' ? 'starting' : 'running')
        }
      } catch (e) {}

      if (active) {
        setTimeout(poll, 1000)
      }
    }

    poll()

    return () => {
      active = false
    }
  }, [classifierRunId])

  const startTraining = async () => {
    setSubmitting(true)
    setError(null)
    setEvents([])
    setStatus('starting')
    try {
      const r = await api.startTraining(cfg)
      if (r.ok && r.data) {
        setRunId(r.data.run_id)
        setModelId(r.data.model_id)
      } else {
        throw new Error(r.detail || 'Failed to start training')
      }
    } catch (e: any) {
      setError(e.message)
      setStatus('idle')
    } finally {
      setSubmitting(false)
    }
  }

  const stopTraining = async () => {
    if (!runId) return
    try {
      await api.stopTrainingRun(runId)
      setStatus('stopping')
    } catch (e: any) {
      setError(e.message)
    }
  }

  const handleUpload = async () => {
    if (!uploadName.trim() || uploadFiles.length === 0) return
    setUploading(true)
    setUploadMsg(null)
    try {
      const r = await api.uploadDataset(uploadName.trim(), uploadFiles)
      if (r.ok) {
        setUploadMsg(`Uploaded ${r.data.saved_files.length} files as "${r.data.id}"`)
        setUploadName('')
        setUploadFiles([])
        api.listDatasets().then(r => setDatasets(r.data))
      }
    } catch (e: any) {
      setUploadMsg(`Upload failed: ${e.message}`)
    } finally {
      setUploading(false)
    }
  }

  const startClassifierTraining = async () => {
    setClassifierStatus('starting')
    setClassifierError(null)
    setClassifierEvents([])
    setClassifierResult(null)
    try {
      const r = await api.startClassifierTraining(classifierCfg)
      if (r.ok && r.data) {
        setClassifierRunId(r.data.run_id)
        setClassifierModelId(r.data.model_id)
      }
    } catch (e: any) {
      setClassifierError(e.message)
      setClassifierStatus('idle')
    }
  }

  // Derived state for charts
  const epochHistory = events.filter(e => e.type === 'epoch') as any[]
  const latestBatch = [...events].reverse().find(e => e.type === 'batch') as any
  const initEvent = events.find(e => e.type === 'init') as any
  const qcEvent = events.find(e => e.type === 'qc') as any
  const finalResult = status === 'completed' && events.length > 0 ? (events[events.length - 1] as any) : null

  // Determine current epoch progress
  const currentEpoch = latestBatch?.epoch || 0
  const totalEpochs = latestBatch?.total_epochs || cfg.epochs
  const epochProgress = latestBatch
    ? ((latestBatch.batch - 1) / latestBatch.total_batches) * 100
    : 0

  return (
    <div className="grid lg:grid-cols-4 gap-6">
      {/* Left column: config + start/stop */}
      <div className="lg:col-span-1 space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-rose-500" />
              Training Configuration
            </CardTitle>
            <CardDescription>
              Configure and launch a new autoencoder training run.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Model name</Label>
              <Input
                value={cfg.model_name}
                onChange={e => setCfg({ ...cfg, model_name: e.target.value })}
              />
            </div>

            <div className="space-y-2">
              <Label>Training dataset</Label>
              <Select
                value={cfg.dataset_name}
                onValueChange={v => setCfg({ ...cfg, dataset_name: v, uploaded_dataset_id: v.startsWith('uploaded:') ? v.substring('uploaded:'.length) : '' })}
              >
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {datasets?.training.map(d => (
                    <SelectItem key={d.id} value={d.id}>
                      <div className="flex flex-col">
                        <span>{d.name}</span>
                        <span className="text-xs text-slate-500">{d.description}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {cfg.dataset_name === 'mit-bih-nsr' && (
                <p className="text-xs text-slate-500">
                  Will use local files in storage/datasets/nsrdb/ if present, otherwise download {cfg.max_records} records from PhysioNet.
                </p>
              )}
              {cfg.dataset_name.startsWith('uploaded:') && (
                <p className="text-xs text-slate-500">
                  Using uploaded dataset "{cfg.uploaded_dataset_id}". {cfg.max_records} records max.
                </p>
              )}
            </div>

            {/* Upload section */}
            <div className="space-y-2 border-t pt-3">
              <Label className="text-xs">Upload your own ECG dataset (.dat + .hea or .csv)</Label>
              <Input
                placeholder="Dataset name (e.g. my-mitbih)"
                value={uploadName}
                onChange={e => setUploadName(e.target.value)}
                className="text-xs h-8"
              />
              <Input
                type="file"
                multiple
                accept=".dat,.hea,.csv"
                onChange={e => setUploadFiles(Array.from(e.target.files || []))}
                className="text-xs h-8 file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:bg-slate-100"
              />
              <Button
                size="sm"
                variant="outline"
                onClick={handleUpload}
                disabled={!uploadName.trim() || uploadFiles.length === 0 || uploading}
                className="w-full h-8 text-xs"
              >
                <Upload className="w-3 h-3 mr-1" />
                {uploading ? 'Uploading...' : 'Upload Dataset'}
              </Button>
              {uploadMsg && (
                <p className={`text-xs ${uploadMsg.includes('failed') ? 'text-rose-600' : 'text-emerald-600'}`}>
                  {uploadMsg}
                </p>
              )}
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Epochs</Label>
                <span className="text-sm font-mono text-slate-600">{cfg.epochs}</span>
              </div>
              <Slider
                min={3} max={100} step={1}
                value={[cfg.epochs]}
                onValueChange={v => setCfg({ ...cfg, epochs: v[0] })}
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label>Batch size</Label>
                <Select
                  value={String(cfg.batch_size)}
                  onValueChange={v => setCfg({ ...cfg, batch_size: Number(v) })}
                >
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {[16, 32, 64, 128].map(n => (
                      <SelectItem key={n} value={String(n)}>{n}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Learning rate</Label>
                <Select
                  value={String(cfg.learning_rate)}
                  onValueChange={v => setCfg({ ...cfg, learning_rate: Number(v) })}
                >
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {[0.0005, 0.001, 0.002].map(n => (
                      <SelectItem key={n} value={String(n)}>{n}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Threshold k (σ multiplier)</Label>
                <span className="text-sm font-mono text-slate-600">{cfg.threshold_k.toFixed(1)}</span>
              </div>
              <Slider
                min={1} max={3} step={0.5}
                value={[cfg.threshold_k]}
                onValueChange={v => setCfg({ ...cfg, threshold_k: v[0] })}
              />
              <p className="text-xs text-slate-500">
                τ = μ + k·σ. Higher k → fewer false positives.
              </p>
            </div>

            {cfg.dataset_name === 'mit-bih-nsr' && (
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Records to download</Label>
                  <span className="text-sm font-mono text-slate-600">{cfg.max_records}</span>
                </div>
                <Slider
                  min={1} max={18} step={1}
                  value={[cfg.max_records]}
                  onValueChange={v => setCfg({ ...cfg, max_records: v[0] })}
                />
              </div>
            )}

            <div className="pt-2">
              {status === 'idle' || status === 'starting' ? (
                <Button
                  onClick={startTraining}
                  disabled={submitting || status === 'starting'}
                  className="w-full bg-rose-600 hover:bg-rose-700"
                >
                  <Play className="w-4 h-4 mr-2" />
                  {submitting ? 'Starting…' : 'Start Training'}
                </Button>
              ) : (
                <Button
                  onClick={stopTraining}
                  disabled={status === 'stopping' || status === 'completed' || status === 'failed'}
                  variant="destructive"
                  className="w-full"
                >
                  <Square className="w-4 h-4 mr-2" />
                  {status === 'stopping' ? 'Stopping…' : 'Stop Training'}
                </Button>
              )}
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertTitle>Training failed</AlertTitle>
                <AlertDescription className="text-xs">{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Architecture preview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Layers className="w-5 h-5 text-rose-500" />
              Current Model Architecture
            </CardTitle>
            <CardDescription>
              Loaded from the backend model code, so Python model changes are reflected here.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <ArchitectureSummary title="Autoencoder" data={architecturePreview?.autoencoder} />
            <ArchitectureSummary title="Classifier" data={architecturePreview?.classifier} />
          </CardContent>
        </Card>

        {/* Classifier Training Section */}
        <Card className="border-amber-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm">
              <Zap className="w-4 h-4 text-amber-500" />
              Train Arrhythmia Classifier (Model 2)
            </CardTitle>
            <CardDescription className="text-xs">
              Uses transfer learning: fine-tunes the Model 1 encoder with a trainable
              morphology branch on MIT-BIH Arrhythmia DB beat annotations.
              Classifies into: N, PVC, PAC, LBBB, RBBB, AFib.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-2">
              <Label className="text-xs">Encoder model (Model 1)</Label>
              <Select
                value={classifierCfg.encoder_model_id}
                onValueChange={v => setClassifierCfg({ ...classifierCfg, encoder_model_id: v })}
                disabled={classifierStatus === 'running' || classifierStatus === 'starting'}
              >
                <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="Select a trained Model 1" /></SelectTrigger>
                <SelectContent>
                  {models.filter(m => m.status === 'ready' && m.model_path).map(m => (
                    <SelectItem key={m.id} value={m.id}>
                      <span className="text-xs">{m.name} ({m.version})</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-xs">Classifier name</Label>
              <Input
                value={classifierCfg.classifier_name}
                onChange={e => setClassifierCfg({ ...classifierCfg, classifier_name: e.target.value })}
                disabled={classifierStatus === 'running' || classifierStatus === 'starting'}
                className="text-xs h-8"
                placeholder="ecg-classifier-v1"
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-2">
                <Label className="text-xs">Epochs</Label>
                <Input
                  type="number"
                  value={classifierCfg.epochs}
                  onChange={e => setClassifierCfg({ ...classifierCfg, epochs: Number(e.target.value) })}
                  disabled={classifierStatus === 'running' || classifierStatus === 'starting'}
                  className="text-xs h-8"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Max records (1-48)</Label>
                <Input
                  type="number"
                  min={1} max={48}
                  value={classifierCfg.max_records}
                  onChange={e => setClassifierCfg({ ...classifierCfg, max_records: Number(e.target.value) })}
                  disabled={classifierStatus === 'running' || classifierStatus === 'starting'}
                  className="text-xs h-8"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-2">
                <Label className="text-xs">Hidden units</Label>
                <Input
                  type="number"
                  min={64}
                  step={64}
                  value={classifierCfg.hidden_dim}
                  onChange={e => setClassifierCfg({ ...classifierCfg, hidden_dim: Number(e.target.value) })}
                  disabled={classifierStatus === 'running' || classifierStatus === 'starting'}
                  className="text-xs h-8"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Dropout</Label>
                <Input
                  type="number"
                  min={0}
                  max={0.8}
                  step={0.05}
                  value={classifierCfg.dropout}
                  onChange={e => setClassifierCfg({ ...classifierCfg, dropout: Number(e.target.value) })}
                  disabled={classifierStatus === 'running' || classifierStatus === 'starting'}
                  className="text-xs h-8"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Encoder LR</Label>
                <Input
                  type="number"
                  min={0}
                  step={0.00005}
                  value={classifierCfg.encoder_learning_rate}
                  onChange={e => setClassifierCfg({ ...classifierCfg, encoder_learning_rate: Number(e.target.value) })}
                  disabled={classifierStatus === 'running' || classifierStatus === 'starting' || classifierCfg.freeze_encoder}
                  className="text-xs h-8"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Focal gamma</Label>
                <Input
                  type="number"
                  min={0}
                  step={0.25}
                  value={classifierCfg.focal_gamma}
                  onChange={e => setClassifierCfg({ ...classifierCfg, focal_gamma: Number(e.target.value) })}
                  disabled={classifierStatus === 'running' || classifierStatus === 'starting'}
                  className="text-xs h-8"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <label className="flex items-center gap-2">
                <Checkbox
                  checked={classifierCfg.augment}
                  onCheckedChange={checked => setClassifierCfg({ ...classifierCfg, augment: checked === true })}
                  disabled={classifierStatus === 'running' || classifierStatus === 'starting'}
                />
                Augment beats
              </label>
              <label className="flex items-center gap-2">
                <Checkbox
                  checked={classifierCfg.freeze_encoder}
                  onCheckedChange={checked => setClassifierCfg({ ...classifierCfg, freeze_encoder: checked === true })}
                  disabled={classifierStatus === 'running' || classifierStatus === 'starting'}
                />
                Freeze encoder
              </label>
            </div>

            <Button
              onClick={startClassifierTraining}
              disabled={!classifierCfg.encoder_model_id || classifierStatus === 'running' || classifierStatus === 'starting'}
              className="w-full bg-amber-600 hover:bg-amber-700 h-8 text-xs"
            >
              <Play className="w-3 h-3 mr-1" />
              {classifierStatus === 'starting' ? 'Starting...' :
               classifierStatus === 'running' ? 'Training...' :
               'Train Classifier'}
            </Button>

            {classifierError && (
              <Alert variant="destructive">
                <AlertDescription className="text-xs">{classifierError}</AlertDescription>
              </Alert>
            )}

            {classifierStatus === 'completed' && classifierResult && (
              <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-xs space-y-1">
                <div className="font-semibold text-emerald-700">Classifier Training Complete!</div>
                <div>Test accuracy: <span className="font-mono font-semibold">{(classifierResult.test_accuracy * 100).toFixed(1)}%</span></div>
                <div>Test macro-F1: <span className="font-mono font-semibold">{((classifierResult.test_macro_f1 ?? 0) * 100).toFixed(1)}%</span></div>
                <div>Best val macro-F1: <span className="font-mono font-semibold">{((classifierResult.val_macro_f1 ?? 0) * 100).toFixed(1)}%</span></div>
                <div>Beats extracted: {classifierResult.n_beats.toLocaleString()}</div>
                <div>Class distribution:</div>
                {Object.entries(classifierResult.class_distribution).map(([cls, count]) => (
                  <div key={cls} className="ml-3 font-mono text-slate-600">{cls}: {String(count)}</div>
                ))}
              </div>
            )}

            {/* Classifier progress events */}
            {classifierEvents.length > 0 && (
              <ScrollArea className="h-32 w-full border rounded-md p-2 bg-slate-950 text-slate-100 font-mono text-xs">
                {classifierEvents.slice(-30).map((ev, i) => (
                  <div key={i} className="mb-0.5">
                    {ev.type === 'loading_encoder' && (
                      <span className="text-cyan-400">Loading encoder checkpoint...</span>
                    )}
                    {ev.type === 'model_built' && (
                      <span className="text-cyan-400">
                        Model built: {ev.architecture?.name} ({ev.architecture?.trainable_parameters?.toLocaleString?.() ?? ev.architecture?.trainable_parameters} trainable params)
                      </span>
                    )}
                    {ev.type === 'gathering_data' && (
                      <span className="text-cyan-400">Gathering MIT-BIH annotated beats...</span>
                    )}
                    {ev.type === 'extraction_progress' && (
                      <span className="text-cyan-400">
                        Records {ev.records_processed}/{ev.total_records}, beats={ev.total_beats}
                      </span>
                    )}
                    {ev.type === 'init' && (
                      <span className="text-cyan-400">
                        Init: train={ev.n_train}, val={ev.n_val}, test={ev.n_test}
                      </span>
                    )}
                    {ev.type === 'epoch' && (
                      <span className="text-yellow-400">
                        EPOCH {ev.epoch}/{ev.total_epochs}: train_acc={ev.train_acc?.toFixed(4)} val_acc={ev.val_acc?.toFixed(4)} val_macro_f1={ev.val_macro_f1?.toFixed(4)}
                      </span>
                    )}
                    {ev.type === 'extraction_complete' && (
                      <span className="text-cyan-400">
                        Extracted {ev.total_beats} beats from MIT-BIH Arrhythmia DB
                      </span>
                    )}
                    {ev.type === 'batch' && (
                      <span className="text-slate-400">batch {ev.batch}/{ev.total_batches} loss={ev.batch_loss?.toFixed(4)}</span>
                    )}
                    {ev.type === 'evaluation' && (
                      <span className="text-emerald-400">
                        Test accuracy: {(ev.test_accuracy * 100).toFixed(1)}%, macro-F1: {(ev.test_macro_f1 * 100).toFixed(1)}%
                      </span>
                    )}
                    {ev.type === 'done' && (
                      <span className="text-emerald-400">DONE: {ev.status}</span>
                    )}
                    {ev.type === 'error' && (
                      <span className="text-rose-400">ERROR: {ev.message}</span>
                    )}
                  </div>
                ))}
              </ScrollArea>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Middle + right columns: live progress */}
      <div className="lg:col-span-3 space-y-4">
        {status === 'idle' && events.length === 0 ? (
          <Card className="border-dashed">
            <CardContent className="py-16 text-center text-slate-500">
              <Activity className="w-12 h-12 mx-auto mb-4 text-slate-300" />
              <p className="text-lg font-medium">No active training run</p>
              <p className="text-sm">Configure parameters on the left and click "Start Training".</p>
            </CardContent>
          </Card>
        ) : (
          <>
            {/* Status banner */}
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    {status === 'completed' && <CheckCircle2 className="w-5 h-5 text-emerald-500" />}
                    {status === 'failed' && <XCircle className="w-5 h-5 text-rose-500" />}
                    {(status === 'running' || status === 'starting') && (
                      <div className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
                    )}
                    <h3 className="font-semibold text-lg">
                      {status === 'idle' && 'Ready'}
                      {status === 'starting' && 'Starting training…'}
                      {status === 'running' && `Training — epoch ${currentEpoch} / ${totalEpochs}`}
                      {status === 'stopping' && 'Stopping…'}
                      {status === 'completed' && 'Training completed'}
                      {status === 'failed' && 'Training failed'}
                    </h3>
                  </div>
                  <Badge variant={
                    status === 'completed' ? 'default' :
                    status === 'failed' ? 'destructive' :
                    status === 'running' ? 'secondary' : 'outline'
                  }>
                    {status}
                  </Badge>
                </div>

                {status === 'running' && latestBatch && (
                  <>
                    <div className="flex justify-between text-xs text-slate-500 mb-1">
                      <span>Epoch {currentEpoch} / {totalEpochs}</span>
                      <span>Batch {latestBatch.batch} / {latestBatch.total_batches} • {latestBatch.elapsed_s}s</span>
                    </div>
                    <Progress value={((currentEpoch - 1) / totalEpochs) * 100 + (epochProgress / totalEpochs)} />
                  </>
                )}

                {initEvent && (
                  <div className="grid grid-cols-3 gap-4 mt-4 text-sm">
                    <Stat label="Train windows" value={initEvent.n_train} icon={<Database className="w-4 h-4" />} />
                    <Stat label="Val windows" value={initEvent.n_val} icon={<Database className="w-4 h-4" />} />
                    <Stat label="Device" value={initEvent.device} icon={<Cpu className="w-4 h-4" />} />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Quality Control stats */}
            {qcEvent && (
              <Card className="border-rose-200 bg-rose-50/30">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Activity className="w-4 h-4 text-rose-500" />
                    ECG Quality Control ({qcEvent.qc_method === 'ecg_qc' ? 'ecg-qc SQI classifier' : 'simple std filter'})
                  </CardTitle>
                  <CardDescription>
                    Windows with quality class ≥ {qcEvent.min_quality} are kept for training.
                    Classes: 0=bad, 1=medium-low, 2=medium-high, 3=excellent.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <Stat
                      label="Windows in"
                      value={qcEvent.n_in ?? qcEvent.n_kept}
                      icon={<Database className="w-4 h-4" />}
                    />
                    <Stat
                      label="Windows kept"
                      value={qcEvent.n_kept}
                      icon={<CheckCircle2 className="w-4 h-4" />}
                    />
                    <Stat
                      label="Retention"
                      value={qcEvent.retention_pct ? `${qcEvent.retention_pct.toFixed(1)}%` : '—'}
                      icon={<TrendingDown className="w-4 h-4" />}
                    />
                    <Stat
                      label="Min quality"
                      value={`≥ ${qcEvent.min_quality}`}
                      icon={<Layers className="w-4 h-4" />}
                    />
                  </div>
                  {qcEvent.distribution && (
                    <div className="mt-4">
                      <div className="text-xs text-slate-500 mb-2">Quality distribution:</div>
                      <div className="flex gap-2 h-8 rounded-md overflow-hidden border border-slate-200">
                        {Object.entries(qcEvent.distribution).map(([cls, count]) => {
                          const total = Object.values(qcEvent.distribution).reduce((a: number, b: any) => a + Number(b), 0)
                          const pct = total > 0 ? (Number(count) / total) * 100 : 0
                          const colors: Record<string, string> = {
                            '0': 'bg-rose-400',
                            '1': 'bg-amber-400',
                            '2': 'bg-emerald-400',
                            '3': 'bg-emerald-600',
                          }
                          const labels: Record<string, string> = {
                            '0': 'Bad',
                            '1': 'Med-low',
                            '2': 'Med-high',
                            '3': 'Excellent',
                          }
                          return pct > 0 ? (
                            <div
                              key={cls}
                              className={`${colors[cls] || 'bg-slate-300'} flex items-center justify-center text-xs font-medium text-white`}
                              style={{ width: `${pct}%` }}
                              title={`${labels[cls]}: ${count} windows (${pct.toFixed(1)}%)`}
                            >
                              {pct > 8 ? `${count}` : ''}
                            </div>
                          ) : null
                        })}
                      </div>
                      <div className="flex gap-2 mt-2 text-xs text-slate-600 flex-wrap">
                        {Object.entries(qcEvent.distribution).map(([cls, count]) => (
                          <span key={cls} className="flex items-center gap-1">
                            <span className={`w-2 h-2 rounded-full ${cls === '0' ? 'bg-rose-400' : cls === '1' ? 'bg-amber-400' : cls === '2' ? 'bg-emerald-400' : 'bg-emerald-600'}`}></span>
                            {cls === '0' ? 'Bad' : cls === '1' ? 'Med-low' : cls === '2' ? 'Med-high' : 'Excellent'}: {String(count)}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Loss curves */}
            {epochHistory.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingDown className="w-5 h-5 text-rose-500" />
                    Loss Curves
                  </CardTitle>
                  <CardDescription>Train vs validation MSE per epoch.</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={epochHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fontSize: 12 }} />
                      <YAxis stroke="#64748b" fontSize={12} label={{ value: 'MSE Loss', angle: -90, position: 'insideLeft', fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12 }}
                        formatter={(v: number) => v.toFixed(6)}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="train_loss" name="Train loss" stroke="#e11d48" strokeWidth={2} dot={{ r: 3 }} />
                      <Line type="monotone" dataKey="val_loss" name="Val loss" stroke="#0891b2" strokeWidth={2} dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}

            {/* Final results */}
            {status === 'completed' && (
              <Card className="border-emerald-200 bg-emerald-50/50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-emerald-700">
                    <CheckCircle2 className="w-5 h-5" />
                    Training Complete
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    {epochHistory.length > 0 && (
                      <>
                        <Stat label="Final train loss" value={epochHistory[epochHistory.length - 1].train_loss.toFixed(6)} />
                        <Stat label="Final val loss" value={epochHistory[epochHistory.length - 1].val_loss.toFixed(6)} />
                        <Stat label="Epochs run" value={epochHistory.length} />
                        <Stat label="Total time" value={`${epochHistory[epochHistory.length - 1].elapsed_s}s`} />
                      </>
                    )}
                  </div>
                  {modelId && (
                    <div className="mt-4 text-sm">
                      <span className="text-slate-500">Model ID: </span>
                      <code className="font-mono text-xs bg-white px-2 py-0.5 rounded border">{modelId}</code>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Live event log */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Live Event Log</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-48 w-full border rounded-md p-3 bg-slate-950 text-slate-100 font-mono text-xs">
                  {events.length === 0 ? (
                    <div className="text-slate-500">Waiting for events…</div>
                  ) : (
                    events.slice(-50).map((ev, i) => (
                      <div key={i} className="mb-1">
                        <span className="text-slate-500">[{new Date().toLocaleTimeString()}]</span>{' '}
                        {ev.type === 'init' && <span className="text-cyan-400">INIT: {ev.n_train} train, {ev.n_val} val windows on {ev.device}</span>}
                        {ev.type === 'qc' && <span className="text-emerald-400">QC: {ev.n_kept}{ev.n_in ? `/${ev.n_in}` : ''} windows kept ({ev.qc_method}, min_quality={ev.min_quality}){ev.distribution ? ` dist=${JSON.stringify(ev.distribution)}` : ''}</span>}
                        {ev.type === 'batch' && <span className="text-slate-400">batch: epoch={ev.epoch}/{ev.total_epochs} batch={ev.batch}/{ev.total_batches} loss={ev.batch_loss.toFixed(6)}</span>}
                        {ev.type === 'epoch' && <span className="text-yellow-400">EPOCH {ev.epoch}/{ev.total_epochs}: train={ev.train_loss.toFixed(6)} val={ev.val_loss.toFixed(6)} ({ev.elapsed_s}s)</span>}
                        {ev.type === 'done' && <span className="text-emerald-400">DONE: {ev.status}</span>}
                      </div>
                    ))
                  )}
                </ScrollArea>
              </CardContent>
            </Card>
          </>
        )}

        {/* Recent training runs */}
        {recentRuns.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Recent Training Runs</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Model</TableHead>
                    <TableHead>Dataset</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Started</TableHead>
                    <TableHead>Final loss</TableHead>
                    <TableHead>Epochs</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {recentRuns.slice(0, 5).map((r: any) => (
                    <TableRow key={r.id}>
                      <TableCell className="font-mono text-xs">{r.model_id?.slice(0, 8)}…</TableCell>
                      <TableCell className="text-xs">{r.dataset}</TableCell>
                      <TableCell>
                        <Badge variant={
                          r.status === 'completed' ? 'default' :
                          r.status === 'failed' ? 'destructive' :
                          r.status === 'running' ? 'secondary' : 'outline'
                        } className="text-xs">
                          {r.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-xs text-slate-500">{r.started_at}</TableCell>
                      <TableCell className="font-mono text-xs">{r.final_loss?.toFixed(6) || '—'}</TableCell>
                      <TableCell className="text-xs">{r.epochs_run || '—'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

function Stat({ label, value, icon }: { label: string; value: any; icon?: React.ReactNode }) {
  return (
    <div className="bg-slate-50 rounded-lg p-3 border">
      <div className="text-xs text-slate-500 flex items-center gap-1 mb-1">
        {icon}
        {label}
      </div>
      <div className="font-mono text-sm font-semibold text-slate-900">{value}</div>
    </div>
  )
}

function ArchitectureSummary({ title, data }: { title: string; data?: any }) {
  if (!data) {
    return (
      <div className="rounded-lg border bg-slate-50 p-3 text-xs text-slate-500">
        Loading {title.toLowerCase()} architecture...
      </div>
    )
  }

  const rows = [
    ['Name', data.name],
    ['Parameters', data.total_parameters?.toLocaleString?.() ?? data.total_parameters],
    ['Input', Array.isArray(data.input_shape) ? `(${data.input_shape.join(', ')})` : data.input_shape],
    ['Latent', Array.isArray(data.latent_shape) ? `(${data.latent_shape.join(', ')})` : data.latent_channels],
    ['Compression', data.compression_ratio ? `${data.compression_ratio}x` : undefined],
    ['Encoder', Array.isArray(data.encoder_channels) ? data.encoder_channels.join(' -> ') : data.encoder],
    ['Decoder', Array.isArray(data.decoder_channels) ? data.decoder_channels.join(' -> ') : undefined],
    ['Kernel / stride', data.kernel_size && data.stride ? `${data.kernel_size} / ${data.stride}` : undefined],
    ['Skip scale', data.skip_scale],
    ['Classes', Array.isArray(data.class_names) ? data.class_names.join(', ') : undefined],
    ['Trainable params', data.trainable_parameters?.toLocaleString?.() ?? data.trainable_parameters],
  ].filter(([, value]) => value !== undefined && value !== null && value !== '')

  return (
    <div className="rounded-lg border bg-slate-50 p-3">
      <div className="mb-2 text-sm font-semibold text-slate-800">{title}</div>
      <div className="space-y-1">
        {rows.map(([label, value]) => (
          <div key={String(label)} className="flex justify-between gap-3 text-xs">
            <span className="text-slate-500">{label}</span>
            <span className="font-mono text-right text-slate-900">{String(value)}</span>
          </div>
        ))}
      </div>
      {Array.isArray(data.features) && (
        <div className="mt-3 border-t pt-2 text-xs text-slate-500">
          {data.features.slice(0, 4).map((feature: string) => (
            <div key={feature}>{feature}</div>
          ))}
        </div>
      )}
    </div>
  )
}
