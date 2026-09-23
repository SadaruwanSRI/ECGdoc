'use client'

import { useEffect, useRef, useState } from 'react'
import { api, wsUrl } from '@/lib/api'
import { EcgCanvas } from '@/components/ecg-canvas'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Play, Square, AlertTriangle, Heart, Activity, Zap, Radio, Database, Server, Bell } from 'lucide-react'
import { io, Socket } from 'socket.io-client'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea,
} from 'recharts'

type ModelInfo = {
  id: string
  name: string
  version: string
  threshold: number | null
  status: string
  architecture?: string
  config_json?: string
  model_path?: string
}

function modelKind(model: ModelInfo): 'classifier' | 'autoencoder' {
  try {
    const cfg = JSON.parse(model.config_json || '{}')
    if (cfg.type === 'classifier') return 'classifier'
  } catch {}
  return String(model.architecture || '').toLowerCase().includes('classifier')
    ? 'classifier'
    : 'autoencoder'
}

function isImprovedHierarchy(model?: ModelInfo): boolean {
  if (!model) return false
  try {
    const cfg = JSON.parse(model.config_json || '{}')
    return cfg.system_kind === 'final_mlii_temporal_holdout'
  } catch {
    return String(model.architecture || '').includes('Experiment2-ExtraTrees-Hierarchy')
  }
}

type SessionPoint = {
  t: number
  value: number
  prediction: number
  anomaly_score: number
  is_anomaly: boolean
}

type AlertItem = {
  id?: string
  timestamp: string | number
  anomaly_score: number
  threshold: number
  severity: string
  message: string
  classification?: {
    class: string
    class_name: string
    confidence: number
    probabilities: Record<string, number>
  } | null
  context?: {
    t?: number
    signal?: number[]
    reconstruction?: number[]
    fs?: number
    score_type?: 'abnormal_probability' | 'reconstruction_error'
    analysis_mode?: string
    rr_seconds?: number
    classification_mode?: string
  }
}

type BeatResult = {
  t: number
  is_anomaly: boolean
  anomaly_probability: number
  threshold: number
  class: string
  class_name: string
  confidence: number
  classification_mode: string
  rr_seconds: number
}

type ChartPoint = {
  t: number
  value: number
  prediction: number
  score: number
  anomaly: boolean
}

type EcgMetrics = {
  bpm: number | null
  hrv_rmssd_ms: number | null
  pr_ms: number | null
  qrs_ms: number | null
  qt_ms: number | null
  qt_corrected_ms: number | null
  rhythm: string
  n_beats: number
  mean_beat: {
    samples: number[]
    fs: number
    n_samples: number
    r_idx: number
    p_onset: number | null
    p_peak: number | null
    qrs_onset: number
    qrs_end: number
    t_peak: number | null
    t_end: number | null
    pr_ms: number | null
    qrs_ms: number
    qt_ms: number | null
  } | null
}

type SerialPortInfo = {
  device: string
  description: string
}

export function LiveTab() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [datasets, setDatasets] = useState<any>(null)
  const [modelId, setModelId] = useState<string>('')
  const [classifierModelId, setClassifierModelId] = useState<string>('__none__')
  const [sourceType, setSourceType] = useState<string>('synthetic-arrhythmia')
  const [sourceDetail, setSourceDetail] = useState<string>('')
  const [chunkSeconds, setChunkSeconds] = useState(4)
  const [useThresholdOverride, setUseThresholdOverride] = useState(false)
  const [thresholdOverride, setThresholdOverride] = useState<number>(0)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [active, setActive] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [healthInfo, setHealthInfo] = useState<any>(null)
  const [socket, setSocket] = useState<Socket | null>(null)
  const [serialPorts, setSerialPorts] = useState<SerialPortInfo[]>([])
  const [sourceStatus, setSourceStatus] = useState<any>(null)
  const [analysisMode, setAnalysisMode] = useState('sliding-reconstruction')
  const [countUnit, setCountUnit] = useState<'beats' | 'samples'>('samples')

  // Live data
  const [chartData, setChartData] = useState<ChartPoint[]>([])
  const [alerts, setAlerts] = useState<AlertItem[]>([])
  const [stats, setStats] = useState({
    totalBeats: 0,
    anomalyBeats: 0,
    threshold: 0,
    avgScore: 0,
    maxScore: 0,
  })
  const [ecgMetrics, setEcgMetrics] = useState<EcgMetrics | null>(null)

  // Refs for chart windowing
  const chartBufferRef = useRef<ChartPoint[]>([])
  const MAX_POINTS = 1024  // ~8 seconds at 128 Hz, downsampled 2x

  useEffect(() => {
    api.listModels().then(r => {
      const list = (r.data?.models || []).filter((m: any) => m.status === 'ready')
      setModels(list)
      const anomalyModels = list.filter((m: ModelInfo) => modelKind(m) === 'autoencoder')
      if (anomalyModels.length > 0) setModelId((anomalyModels.find((m: ModelInfo) => m.id === 'nsrdb-primary-autoencoder') || anomalyModels[0]).id)
      const improved = list.find((m: ModelInfo) => m.id === 'final-mlii-corrected-20260916') || list.find((m: ModelInfo) => isImprovedHierarchy(m))
      if (improved) setClassifierModelId(improved.id)
    })
    api.listDatasets().then(r => setDatasets(r.data))
    api.listArduinoPorts().then(r => {
      const ports = r.data?.ports || []
      setSerialPorts(ports)
      if (ports.length === 1) setSourceDetail(ports[0].device)
    }).catch(() => setSerialPorts([]))
  }, [])

  useEffect(() => {
    const selectedClassifier = models.find(m => m.id === classifierModelId)
    const selected = isImprovedHierarchy(selectedClassifier)
      ? selectedClassifier
      : models.find(m => m.id === modelId)
    if (selected?.threshold != null) {
      setThresholdOverride(Number(selected.threshold))
    }
  }, [modelId, classifierModelId, models])

  useEffect(() => {
    if (sourceType === 'arduino') {
      setSourceDetail(current => current || serialPorts[0]?.device || '')
    } else if (sourceType === 'mit-bih-arrhythmia') {
      setSourceDetail('100')
    } else if (sourceType === 'incartdb') {
      setSourceDetail('I01')
    } else {
      setSourceDetail('')
    }
  }, [sourceType, serialPorts])

  useEffect(() => {
    if (!sessionId) return
    const sock = io(wsUrl(), {
      transports: ['websocket', 'polling'],
      forceNew: true,
    })
    setSocket(sock)

    sock.on('connect', () => {
      const token = localStorage.getItem('ecg_token') || ''
      sock.emit('subscribe:session', { session_id: sessionId, token })
    })

    sock.on('chunk', (data: any) => {
      if (data.type !== 'chunk') return
      const points: SessionPoint[] = data.points || []
      const newAlerts: AlertItem[] = data.alerts || []
      const beatResults: BeatResult[] = data.beat_results || []

      // Convert to chart points
      const newChartPoints: ChartPoint[] = points.map(p => ({
        t: p.t,
        value: p.value,
        prediction: p.prediction,
        score: p.anomaly_score,
        anomaly: p.is_anomaly,
      }))

      // Rolling buffer: keep last MAX_POINTS (a sliding window, NOT growing)
      const combined = [...chartBufferRef.current, ...newChartPoints].slice(-MAX_POINTS)
      for (const beat of beatResults) {
        for (const point of combined) {
          if (Math.abs(point.t - beat.t) <= 10) {
            point.score = beat.anomaly_probability
            point.anomaly = beat.is_anomaly
          }
        }
      }
      chartBufferRef.current = combined
      // Create a NEW array reference so React detects the change and re-renders cleanly
      setChartData(chartBufferRef.current)

      if (newAlerts.length > 0) {
        setAlerts(prev => [...newAlerts, ...prev].slice(0, 50))
      }

      // Update ECG vital signs metrics (BPM, intervals, mean beat)
      if (data.ecg_metrics) {
        setEcgMetrics(data.ecg_metrics)
      }
      if (data.source_status) setSourceStatus(data.source_status)
      if (data.analysis_mode) setAnalysisMode(data.analysis_mode)
      if (data.count_unit) setCountUnit(data.count_unit)

      const activeScores = beatResults.length > 0
        ? beatResults.map(beat => beat.anomaly_probability)
        : newChartPoints.map(point => point.score)

      setStats({
        totalBeats: data.total_beats || 0,
        anomalyBeats: data.anomaly_beats || 0,
        threshold: data.threshold || 0,
        avgScore: activeScores.length > 0
          ? activeScores.reduce((sum, score) => sum + score, 0) / activeScores.length
          : 0,
        maxScore: activeScores.length > 0
          ? Math.max(...activeScores)
          : 0,
      })
    })

    sock.on('end', () => {
      setActive(false)
    })

    sock.on('error', (e: any) => {
      console.error('ws error', e)
      setError(e.message || 'Session error')
      setActive(false)
    })

    return () => {
      sock.disconnect()
    }
  }, [sessionId])

  const start = async () => {
    setError(null)
    setChartData([])
    setAlerts([])
    setEcgMetrics(null)
    setSourceStatus(null)
    setAnalysisMode('sliding-reconstruction')
    setCountUnit('samples')
    chartBufferRef.current = []
    setStats({ totalBeats: 0, anomalyBeats: 0, threshold: 0, avgScore: 0, maxScore: 0 })
    try {
      const r = await api.startSession({
        model_id: modelId,
        classifier_model_id: classifierModelId === '__none__' ? undefined : classifierModelId,
        source_type: sourceType,
        source_detail: sourceDetail || undefined,
        lead_name: 'MLII',
        chunk_seconds: sourceType === 'arduino' ? 4 : chunkSeconds,
        threshold_override: useThresholdOverride ? thresholdOverride : undefined,
      })
      if (r.ok && r.data?.session_id) {
        setSessionId(r.data.session_id)
        setAnalysisMode(r.data.analysis_mode || r.data.model_info?.analysis_mode || 'sliding-reconstruction')
        setCountUnit(r.data.count_unit || 'samples')
        setStats(s => ({
          ...s,
          threshold: r.data.model_info?.decision_threshold
            ?? r.data.model_info?.threshold
            ?? 0,
        }))
        setActive(true)
      } else {
        throw new Error(r.detail || 'Failed to start session')
      }
    } catch (e: any) {
      setError(e.message)
    }
  }

  const stop = async () => {
    if (!sessionId) return
    try {
      await api.stopSession(sessionId)
    } catch (e) {
      console.error(e)
    }
    setActive(false)
    socket?.disconnect()
  }

  const anomalyPct = stats.totalBeats > 0 ? (stats.anomalyBeats / stats.totalBeats) * 100 : 0
  const currentSource = datasets?.testing.find((d: any) => d.id === sourceType)
  const selectedClassifier = models.find(model => model.id === classifierModelId)
  const usingImprovedSystem = isImprovedHierarchy(selectedClassifier)
  const selectedThresholdModel = usingImprovedSystem
    ? selectedClassifier
    : models.find(model => model.id === modelId)

  return (
    <div className="grid lg:grid-cols-5 gap-6">
      {/* Left: config */}
      <div className="lg:col-span-1 space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Radio className="w-5 h-5 text-rose-500" />
              Live Analysis Setup
            </CardTitle>
            <CardDescription>Configure data stream & model.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>
                {usingImprovedSystem
                  ? 'Waveform reconstruction model (display fallback)'
                  : 'Anomaly detection model (Model 1)'}
              </Label>
              <Select value={modelId} onValueChange={setModelId} disabled={active}>
                <SelectTrigger><SelectValue placeholder="Select trained model" /></SelectTrigger>
                <SelectContent>
                  {models.length === 0 && (
                    <SelectItem value="__none__" disabled>No ready models — train one first</SelectItem>
                  )}
                  {models.filter(m => modelKind(m) === 'autoencoder').map(m => (
                    <SelectItem key={m.id} value={m.id}>
                      <div className="flex flex-col">
                        <span>{m.name}</span>
                        <span className="text-xs text-slate-500 font-mono">{m.version} • τ={m.threshold?.toFixed(3)}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Detection and rhythm model</Label>
              <Select value={classifierModelId} onValueChange={setClassifierModelId} disabled={active}>
                <SelectTrigger><SelectValue placeholder="None (anomaly detection only)" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">None (anomaly detection only)</SelectItem>
                  {models.filter(m => modelKind(m) === 'classifier').map(m => (
                    <SelectItem key={m.id} value={m.id}>
                      <div className="flex flex-col">
                        <span>{m.name}</span>
                        <span className="text-xs text-slate-500 font-mono">{m.version}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-slate-500">
                {usingImprovedSystem
                  ? 'Final system: R-peak-aligned morphology and RR timing detect abnormal beats; the subtype stage names supported rhythms.'
                  : 'Optional legacy classifier for anomalies detected by reconstruction error.'}
              </p>
              {usingImprovedSystem && (
                <Alert className="border-emerald-200 bg-emerald-50">
                  <Activity className="h-4 w-4 text-emerald-700" />
                  <AlertTitle className="text-xs text-emerald-900">Improved beat-aligned mode</AlertTitle>
                  <AlertDescription className="text-xs text-emerald-800">
                    The complete system accepts MLII only. MIT–BIH replay selects MLII
                    by name; Arduino electrodes must follow the MLII torso placement.
                    Predictions are delayed until the next RR
                    interval and post-peak waveform are available.
                  </AlertDescription>
                </Alert>
              )}
            </div>

            <div className="space-y-2">
              <Label>Data source</Label>
              <Select value={sourceType} onValueChange={setSourceType} disabled={active}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {datasets?.testing.map((d: any) => (
                    <SelectItem key={d.id} value={d.id}>
                      <div className="flex flex-col">
                        <span className="flex items-center gap-1.5">
                          {d.id === 'arduino' ? <Radio className="w-3 h-3" /> :
                           d.id === 'mit-bih-arrhythmia' ? <Database className="w-3 h-3" /> :
                           <Activity className="w-3 h-3" />}
                          {d.name}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {currentSource && (
                <p className="text-xs text-slate-500">{currentSource.description}</p>
              )}
            </div>

            {(sourceType === 'mit-bih-arrhythmia' || sourceType === 'incartdb') && (
              <div className="space-y-2">
                <Label>Record name</Label>
                <Select
                  value={sourceDetail || (sourceType === 'incartdb' ? 'I01' : '100')}
                  onValueChange={setSourceDetail}
                  disabled={active}
                >
                  <SelectTrigger><SelectValue placeholder="Pick a record" /></SelectTrigger>
                  <SelectContent>
                    {(datasets?.testing.find((d: any) => d.id === sourceType)?.records || []).map((r: string) => (
                      <SelectItem key={r} value={r}>{r}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-slate-500">
                  {sourceType === 'incartdb'
                    ? 'Standard lead II is replayed as an external-domain counterpart to the model’s MLII input.'
                    : 'Only records containing MLII are listed; MLII is selected by name.'}
                </p>
              </div>
            )}

            {sourceType === 'arduino' && (
              <div className="space-y-2">
                <Label>Arduino serial port</Label>
                <Select value={sourceDetail} onValueChange={setSourceDetail} disabled={active}>
                  <SelectTrigger><SelectValue placeholder="Select COM port" /></SelectTrigger>
                  <SelectContent>
                    {serialPorts.map(port => (
                      <SelectItem key={port.device} value={port.device}>
                        {port.device} — {port.description || 'Serial device'}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {serialPorts.length === 0 && (
                  <p className="text-xs text-amber-600">No serial ports found. Connect the Nano and restart the backend.</p>
                )}
                <p className="text-xs text-amber-700">
                  Required placement: MLII, with the negative electrode on the upper-right torso and the positive electrode toward the lower-left torso.
                </p>
                {sourceStatus && (
                  <p className={`text-xs ${sourceStatus.lead_off ? 'text-amber-600' : 'text-emerald-600'}`}>
                    {sourceStatus.lead_off
                      ? 'Electrodes disconnected — streaming raw hardware data.'
                      : 'Electrodes connected'}
                  </p>
                )}
              </div>
            )}

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Chunk size (s)</Label>
                <span className="text-sm font-mono text-slate-600">{(sourceType === 'arduino' ? 4 : chunkSeconds).toFixed(1)}s</span>
              </div>
              <Slider
                min={1} max={8} step={0.5}
                value={[chunkSeconds]}
                onValueChange={v => setChunkSeconds(v[0])}
                disabled={active || sourceType === 'arduino'}
              />
              {sourceType === 'arduino' && (
                <p className="text-xs text-slate-500">Fixed at 4 seconds (512 samples) for the trained model.</p>
              )}
            </div>

            <div className="space-y-2 rounded-md border bg-slate-50 p-3">
              <div className="flex items-center justify-between gap-2">
                <Label>
                  {usingImprovedSystem
                    ? 'Abnormal-probability threshold'
                    : 'Reconstruction-error threshold'}
                </Label>
                <Button
                  type="button"
                  size="sm"
                  variant={useThresholdOverride ? 'default' : 'outline'}
                  onClick={() => setUseThresholdOverride(v => !v)}
                  disabled={active}
                  className="h-7 text-xs"
                >
                  {useThresholdOverride ? 'Custom' : 'Model default'}
                </Button>
              </div>
              <div className="text-xs text-slate-500">
                {usingImprovedSystem
                  ? 'Probability above this value is abnormal. The saved value was selected on validation records by balanced accuracy.'
                  : 'Lower threshold = more sensitive. Higher threshold = fewer alerts.'}
              </div>
              <Input
                type="number"
                step="0.001"
                min="0"
                value={thresholdOverride}
                onChange={e => setThresholdOverride(Number(e.target.value))}
                disabled={active || !useThresholdOverride}
                className="h-8 font-mono text-xs"
              />
              <div className="flex justify-between text-xs text-slate-500">
                <span>Saved: {selectedThresholdModel?.threshold?.toFixed(4) ?? '-'}</span>
                <span>Using: {(useThresholdOverride ? thresholdOverride : selectedThresholdModel?.threshold || 0).toFixed(4)}</span>
              </div>
            </div>

            <div className="pt-2">
              {!active ? (
                <Button
                  onClick={start}
                  disabled={!modelId || (sourceType === 'arduino' && !sourceDetail)}
                  className="w-full bg-rose-600 hover:bg-rose-700"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Start Live Analysis
                </Button>
              ) : (
                <Button onClick={stop} variant="destructive" className="w-full">
                  <Square className="w-4 h-4 mr-2" />
                  Stop Session
                </Button>
              )}
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertDescription className="text-xs">{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Quick diagnostics */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <Zap className="w-4 h-4 text-rose-500" />
              Diagnostics
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <DiagRow label="Source" value={sourceType} icon={<Database className="w-3 h-3" />} />
            <DiagRow label="Status" value={active ? 'STREAMING' : 'IDLE'} icon={<Activity className="w-3 h-3" />} highlight={active} />
            <DiagRow label="Analysis" value={analysisMode === 'beat-aligned-hierarchical' ? 'BEAT + RR' : 'RECONSTRUCTION'} icon={<Server className="w-3 h-3" />} highlight={analysisMode === 'beat-aligned-hierarchical'} />
            <DiagRow label="Heart Rate" value={ecgMetrics?.bpm != null ? `${ecgMetrics.bpm.toFixed(0)} BPM` : '—'} icon={<Heart className="w-3 h-3" />} warn={ecgMetrics?.bpm != null && (ecgMetrics.bpm < 60 || ecgMetrics.bpm > 100)} />
            <DiagRow label="Rhythm" value={ecgMetrics?.rhythm || '—'} icon={<Activity className="w-3 h-3" />} warn={Boolean(ecgMetrics?.rhythm && !ecgMetrics.rhythm.includes('Normal'))} />
            <DiagRow label={usingImprovedSystem ? 'Probability threshold' : 'MAE threshold'} value={stats.threshold.toFixed(4)} icon={<Server className="w-3 h-3" />} />
            <DiagRow label={usingImprovedSystem ? 'Latest beat probability' : 'Current score'} value={stats.maxScore.toFixed(4)} icon={<Zap className="w-3 h-3" />} warn={stats.maxScore > stats.threshold && stats.threshold > 0} />
            <DiagRow label={`${countUnit === 'beats' ? 'Beats' : 'Samples'} analyzed`} value={stats.totalBeats.toLocaleString()} icon={<Heart className="w-3 h-3" />} />
            <DiagRow label={`Anomalous ${countUnit}`} value={stats.anomalyBeats.toLocaleString()} icon={<AlertTriangle className="w-3 h-3" />} warn={stats.anomalyBeats > 0} />
            <DiagRow label="Anomaly rate" value={`${anomalyPct.toFixed(2)}%`} icon={<Bell className="w-3 h-3" />} warn={anomalyPct > 1} />
          </CardContent>
        </Card>
      </div>

      {/* Middle + right: live charts & alerts */}
      <div className="lg:col-span-4 space-y-4">
        {/* ECG waveform */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Heart className="w-5 h-5 text-rose-500" />
                  Live ECG Signal
                </CardTitle>
                <CardDescription className="text-xs">
                  {active
                    ? `Streaming • ${stats.totalBeats} ${countUnit} analyzed`
                    : 'Idle — click Start to begin'}
                </CardDescription>
              </div>
              {active && (
                <div className="flex items-center gap-2 text-sm">
                  <span className="relative flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-rose-500"></span>
                  </span>
                  <span className="font-mono">LIVE</span>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <EcgCanvas
              data={chartData.map(p => ({
                t: p.t,
                value: p.value,
                prediction: p.prediction,
                anomaly: p.anomaly,
              }))}
              active={active}
              height={300}
              pixelsPerSample={3}
              showPrediction={!usingImprovedSystem}
            />
            {/* Legend */}
            <div className="flex items-center justify-center gap-4 mt-3 text-xs text-slate-500">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-slate-900"></span>ECG signal
              </span>
              {!usingImprovedSystem && (
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-cyan-600 border-t border-dashed"></span>Reconstruction
                </span>
              )}
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 bg-rose-200"></span>Anomaly region
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-green-500"></span>Sweep cursor
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Anomaly score chart */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <Activity className="w-4 h-4 text-rose-500" />
              {usingImprovedSystem
                ? 'Beat Abnormal Probability vs Threshold'
                : 'Anomaly Score (MAE) vs Threshold'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={140} key={`score-${chartData.length > 0 ? chartData[chartData.length - 1].t : 0}`}>
              <LineChart data={chartData} syncId="live-ecg">
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="t"
                  stroke="#64748b"
                  fontSize={10}
                  type="number"
                  domain={['dataMin', 'dataMax']}
                  tickFormatter={(v) => `${(v / 128).toFixed(1)}s`}
                />
                <YAxis stroke="#64748b" fontSize={10} />
                <Tooltip
                  contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 11 }}
                  formatter={(v: number) => v?.toFixed(4)}
                />
                {stats.threshold > 0 && (
                  <ReferenceLine y={stats.threshold} stroke="#dc2626" strokeWidth={1.5} strokeDasharray="4 4"
                    label={{ value: `t=${stats.threshold.toFixed(3)}`, fill: '#dc2626', fontSize: 10, position: 'right' }} />
                )}
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="#e11d48"
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* ECG Vital Signs + Mean Beat */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <Heart className="w-4 h-4 text-rose-500" />
              ECG Vital Signs {ecgMetrics?.rhythm && (
                <Badge variant={ecgMetrics.rhythm.includes('Normal') ? 'default' : 'destructive'} className="text-xs ml-1">
                  {ecgMetrics.rhythm}
                </Badge>
              )}
            </CardTitle>
            <CardDescription className="text-xs">
              Real-time cardiac metrics from R-peak detection. Updated every ~5 seconds.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-4">
              {/* Vital signs tiles */}
              <div className="grid grid-cols-2 gap-3">
                <VitalTile
                  label="Heart Rate"
                  value={ecgMetrics?.bpm != null ? ecgMetrics.bpm.toFixed(0) : '—'}
                  unit="BPM"
                  normal={ecgMetrics?.bpm != null && ecgMetrics.bpm >= 60 && ecgMetrics.bpm <= 100}
                  warn={ecgMetrics?.bpm != null && (ecgMetrics.bpm < 60 || ecgMetrics.bpm > 100)}
                  hint="Normal: 60-100"
                />
                <VitalTile
                  label="HRV (RMSSD)"
                  value={ecgMetrics?.hrv_rmssd_ms != null ? ecgMetrics.hrv_rmssd_ms.toFixed(0) : '—'}
                  unit="ms"
                  normal={ecgMetrics?.hrv_rmssd_ms != null && ecgMetrics.hrv_rmssd_ms >= 20 && ecgMetrics.hrv_rmssd_ms <= 100}
                  warn={ecgMetrics?.hrv_rmssd_ms != null && ecgMetrics.hrv_rmssd_ms > 100}
                  hint="Normal: 20-100ms"
                />
                <VitalTile
                  label="PR Interval"
                  value={ecgMetrics?.pr_ms != null ? ecgMetrics.pr_ms.toFixed(0) : '—'}
                  unit="ms"
                  normal={ecgMetrics?.pr_ms != null && ecgMetrics.pr_ms >= 120 && ecgMetrics.pr_ms <= 200}
                  warn={ecgMetrics?.pr_ms != null && (ecgMetrics.pr_ms < 120 || ecgMetrics.pr_ms > 200)}
                  hint="Normal: 120-200ms"
                />
                <VitalTile
                  label="QRS Duration"
                  value={ecgMetrics?.qrs_ms != null ? ecgMetrics.qrs_ms.toFixed(0) : '—'}
                  unit="ms"
                  normal={ecgMetrics?.qrs_ms != null && ecgMetrics.qrs_ms >= 80 && ecgMetrics.qrs_ms <= 120}
                  warn={ecgMetrics?.qrs_ms != null && (ecgMetrics.qrs_ms < 80 || ecgMetrics.qrs_ms > 120)}
                  hint="Normal: 80-120ms"
                />
                <VitalTile
                  label="QT Interval"
                  value={ecgMetrics?.qt_ms != null ? ecgMetrics.qt_ms.toFixed(0) : '—'}
                  unit="ms"
                  normal={ecgMetrics?.qt_ms != null && ecgMetrics.qt_ms >= 350 && ecgMetrics.qt_ms <= 450}
                  warn={ecgMetrics?.qt_ms != null && (ecgMetrics.qt_ms < 350 || ecgMetrics.qt_ms > 450)}
                  hint="Normal: 350-450ms"
                />
                <VitalTile
                  label="QTc (Bazett)"
                  value={ecgMetrics?.qt_corrected_ms != null ? ecgMetrics.qt_corrected_ms.toFixed(0) : '—'}
                  unit="ms"
                  normal={ecgMetrics?.qt_corrected_ms != null && ecgMetrics.qt_corrected_ms >= 350 && ecgMetrics.qt_corrected_ms <= 440}
                  warn={ecgMetrics?.qt_corrected_ms != null && (ecgMetrics.qt_corrected_ms < 350 || ecgMetrics.qt_corrected_ms > 440)}
                  hint="Normal: 350-440ms"
                />
              </div>

              {/* Mean beat chart */}
              <div>
                <div className="text-xs text-slate-500 mb-2">Mean ECG Beat (one cardiac cycle)</div>
                {ecgMetrics?.mean_beat ? (
                  <ResponsiveContainer width="100%" height={180}>
                    <LineChart
                      data={ecgMetrics.mean_beat.samples.map((v, i) => ({
                        idx: i,
                        value: v,
                        p_onset: ecgMetrics.mean_beat!.p_onset,
                        qrs_onset: ecgMetrics.mean_beat!.qrs_onset,
                        qrs_end: ecgMetrics.mean_beat!.qrs_end,
                        t_end: ecgMetrics.mean_beat!.t_end,
                      }))}
                      margin={{ top: 8, right: 16, bottom: 4, left: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis
                        dataKey="idx"
                        stroke="#64748b"
                        fontSize={9}
                        type="number"
                        domain={['dataMin', 'dataMax']}
                        tickFormatter={(v) => `${(v / ecgMetrics.mean_beat!.fs * 1000).toFixed(0)}ms`}
                      />
                      <YAxis stroke="#64748b" fontSize={9} domain={['auto', 'auto']} />
                      <Tooltip
                        contentStyle={{ borderRadius: 6, border: '1px solid #e2e8f0', fontSize: 10 }}
                        labelFormatter={(v) => `${(Number(v) / ecgMetrics.mean_beat!.fs * 1000).toFixed(0)}ms`}
                        formatter={(v: number) => [v?.toFixed(3), 'Amplitude']}
                      />
                      {/* P-wave region */}
                      {ecgMetrics.mean_beat.p_onset != null && (
                        <ReferenceArea
                          x1={ecgMetrics.mean_beat.p_onset}
                          x2={ecgMetrics.mean_beat.qrs_onset}
                          fill="#3b82f6"
                          fillOpacity={0.15}
                        />
                      )}
                      {/* QRS region */}
                      <ReferenceArea
                        x1={ecgMetrics.mean_beat.qrs_onset}
                        x2={ecgMetrics.mean_beat.qrs_end}
                        fill="#dc2626"
                        fillOpacity={0.2}
                      />
                      {/* QT region (QRS onset to T end) */}
                      {ecgMetrics.mean_beat.t_end != null && (
                        <ReferenceArea
                          x1={ecgMetrics.mean_beat.qrs_onset}
                          x2={ecgMetrics.mean_beat.t_end}
                          fill="#a855f7"
                          fillOpacity={0.08}
                        />
                      )}
                      {/* R-peak marker */}
                      <ReferenceLine x={ecgMetrics.mean_beat.r_idx} stroke="#dc2626" strokeWidth={1} strokeDasharray="2 2" />
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke="#0f172a"
                        strokeWidth={1.5}
                        dot={false}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-[180px] flex items-center justify-center text-sm text-slate-400 border rounded-md bg-slate-50">
                    {active ? 'Analyzing heartbeats...' : 'Start a session to compute mean beat'}
                  </div>
                )}
                {/* Legend for intervals */}
                <div className="flex items-center gap-3 mt-2 text-xs text-slate-600 flex-wrap">
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 bg-blue-500 opacity-50"></span>PR
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 bg-rose-500 opacity-50"></span>QRS
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 bg-purple-500 opacity-30"></span>QT
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-0.5 bg-rose-600"></span>R-peak
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Detection Result + Interpretation */}
        {alerts.length > 0 && (
          <DetectionResultCard alerts={alerts} onGetHealthInfo={setHealthInfo} />
        )}
        {healthInfo && (
          <InterpretationCard info={healthInfo} onClose={() => setHealthInfo(null)} />
        )}

        {/* Alerts panel */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm flex items-center gap-2">
                <Bell className="w-4 h-4 text-rose-500" />
                Alerts ({alerts.length})
              </CardTitle>
              {alerts.length > 0 && (
                <Badge variant="destructive">{alerts.filter(a => a.severity === 'critical').length} critical</Badge>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-80 w-full border rounded-md">
              {alerts.length === 0 ? (
                <div className="text-sm text-slate-500 text-center py-8">
                  No alerts. The model is classifying the ECG as within normal limits.
                </div>
              ) : (
                <div className="divide-y">
                  {alerts.map((a, i) => {
                    const ctx = a.context || {}
                    const hasSignal = ctx.signal && ctx.signal.length > 0
                    const signalData = hasSignal
                      ? ctx.signal!.map((v, idx) => ({
                          t: idx,
                          value: v,
                          recon: ctx.reconstruction?.[idx] ?? null,
                        }))
                      : []
                    return (
                      <div key={i} className={`p-3 ${
                        a.severity === 'critical' ? 'bg-rose-50' : 'bg-amber-50'
                      }`}>
                        <div className="flex items-start gap-3">
                          <AlertTriangle className={`w-4 h-4 mt-0.5 shrink-0 ${
                            a.severity === 'critical' ? 'text-rose-600' : 'text-amber-600'
                          }`} />
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-medium">{a.message}</div>
                            <div className="text-xs text-slate-500 mt-0.5">
                              {ctx.score_type === 'abnormal_probability'
                                ? `abnormal probability=${(a.anomaly_score * 100).toFixed(1)}%`
                                : `score=${a.anomaly_score.toFixed(4)}`}
                              {' • '}threshold={a.threshold.toFixed(4)}{' • '}
                              {typeof a.timestamp === 'number'
                                ? new Date(a.timestamp * 1000).toLocaleTimeString()
                                : new Date(a.timestamp).toLocaleTimeString()}
                            </div>
                          </div>
                          <Badge variant={a.severity === 'critical' ? 'destructive' : 'default'} className="text-xs shrink-0">
                            {a.severity}
                          </Badge>
                        </div>
                        {hasSignal && (
                          <div className="mt-2 ml-7 h-20 bg-white rounded border border-slate-200 overflow-hidden">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={signalData} margin={{ top: 2, right: 4, bottom: 0, left: 0 }}>
                                <CartesianGrid strokeDasharray="2 4" stroke="#e2e8f0" />
                                <XAxis dataKey="t" hide />
                                <YAxis hide domain={['auto', 'auto']} />
                                <Tooltip
                                  contentStyle={{ borderRadius: 6, border: '1px solid #e2e8f0', fontSize: 10, padding: '4px 8px' }}
                                  labelFormatter={(_, payload) => {
                                    if (!payload?.[0]) return ''
                                    const fs = ctx.fs || 64
                                    return `t=${(payload[0].payload.t / fs).toFixed(2)}s`
                                  }}
                                  formatter={(v: number, name: string) => [
                                    v?.toFixed(3) ?? '—',
                                    name === 'value' ? 'ECG' : 'Reconstruction',
                                  ]}
                                />
                                <Line
                                  type="monotone"
                                  dataKey="value"
                                  stroke="#dc2626"
                                  strokeWidth={1.2}
                                  dot={false}
                                  isAnimationActive={false}
                                />
                                <Line
                                  type="monotone"
                                  dataKey="recon"
                                  stroke="#0891b2"
                                  strokeWidth={1}
                                  strokeDasharray="3 2"
                                  dot={false}
                                  isAnimationActive={false}
                                  connectNulls
                                />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        )}
                        {hasSignal && (
                          <div className="mt-1 ml-7 flex items-center gap-3 text-xs text-slate-500">
                            <span className="flex items-center gap-1">
                              <span className="w-2.5 h-0.5 bg-rose-600"></span>
                              ECG (anomalous)
                            </span>
                            <span className="flex items-center gap-1">
                              <span className="w-2.5 h-0.5 bg-cyan-600 border-t border-dashed"></span>
                              Reconstruction
                            </span>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function DiagRow({ label, value, icon, warn, highlight }: {
  label: string
  value: string
  icon?: React.ReactNode
  warn?: boolean
  highlight?: boolean
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-slate-500 flex items-center gap-1.5 text-xs">
        {icon}
        {label}
      </span>
      <span className={`font-mono text-sm ${
        warn ? 'text-rose-600 font-semibold' :
        highlight ? 'text-rose-600 font-semibold' :
        'text-slate-900'
      }`}>
        {value}
      </span>
    </div>
  )
}

function VitalTile({ label, value, unit, normal, warn, hint }: {
  label: string
  value: string
  unit: string
  normal?: boolean
  warn?: boolean
  hint?: string
}) {
  return (
    <div className={`rounded-lg border p-3 transition-colors ${
      normal ? 'border-emerald-200 bg-emerald-50' :
      warn ? 'border-rose-200 bg-rose-50' :
      'border-slate-200 bg-slate-50'
    }`}>
      <div className="text-xs text-slate-500 mb-1">{label}</div>
      <div className="flex items-baseline gap-1">
        <span className={`text-2xl font-bold font-mono ${
          normal ? 'text-emerald-700' :
          warn ? 'text-rose-700' :
          'text-slate-700'
        }`}>
          {value}
        </span>
        <span className="text-xs text-slate-500">{unit}</span>
      </div>
      {hint && <div className="text-xs text-slate-400 mt-1">{hint}</div>}
    </div>
  )
}


// Detection Result Card — shows the latest anomaly detection + classification
function DetectionResultCard({ alerts, onGetHealthInfo }: {
  alerts: AlertItem[]
  onGetHealthInfo: (info: any) => void
}) {
  const latestAlert = alerts[0]
  const [showHealthInfo, setShowHealthInfo] = useState(false)

  if (!latestAlert) return null

  const isAnomaly = true  // alerts only appear for anomalies
  const classification = latestAlert.classification
  const probabilityMode = latestAlert.context?.score_type === 'abnormal_probability'

  const handleGetInfo = async () => {
    if (!classification) return
    try {
      const r = await api.getHealthInfo(classification.class)
      if (r.ok) onGetHealthInfo(r.data)
    } catch (e) { console.error(e) }
  }

  return (
    <Card className={isAnomaly ? 'border-rose-300' : 'border-emerald-300'}>
      <CardHeader>
        <CardTitle className="text-sm flex items-center gap-2">
          <Activity className="w-4 h-4 text-rose-500" />
          Detection Result
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Step 1: Anomaly Detection */}
        <div className={`rounded-lg border p-3 ${isAnomaly ? 'border-rose-200 bg-rose-50' : 'border-emerald-200 bg-emerald-50'}`}>
          <div className="flex items-center gap-2 mb-1">
            <AlertTriangle className={`w-4 h-4 ${isAnomaly ? 'text-rose-600' : 'text-emerald-600'}`} />
            <span className="font-medium text-sm">
              Step 1: {isAnomaly ? 'Abnormal Beat Detected' : 'Normal'}
            </span>
          </div>
          <div className="text-xs text-slate-600">
            {probabilityMode ? 'Abnormal probability' : 'Reconstruction error'}:{' '}
            {probabilityMode
              ? `${(latestAlert.anomaly_score * 100).toFixed(1)}%`
              : latestAlert.anomaly_score.toFixed(4)}
            {' '}(threshold: {latestAlert.threshold.toFixed(4)})
          </div>
        </div>

        {/* Step 2: Arrhythmia Classification */}
        {classification ? (
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-amber-600" />
              <span className="font-medium text-sm">Step 2: Rhythm Subtype</span>
            </div>
            <div className="text-lg font-bold text-slate-900 mb-1">
              {classification.class_name}
            </div>
            <div className="text-xs text-slate-600 mb-2">
              Confidence: {(classification.confidence * 100).toFixed(1)}%
            </div>
            {/* Probability bars */}
            <div className="space-y-1">
              {Object.entries(classification.probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([cls, prob]) => (
                  <div key={cls} className="flex items-center gap-2 text-xs">
                    <span className="w-12 text-slate-500 font-mono">{cls}</span>
                    <div className="flex-1 h-3 bg-slate-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${cls === classification.class ? 'bg-rose-500' : 'bg-slate-400'}`}
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <span className="w-10 text-right font-mono text-slate-600">{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
            </div>
            {classification.class !== 'Unclassified abnormal' && (
              <Button
                size="sm"
                variant="outline"
                onClick={handleGetInfo}
                className="mt-3 w-full"
              >
                <Heart className="w-3 h-3 mr-1" />
                What does this mean? (Health Info)
              </Button>
            )}
          </div>
        ) : (
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="text-xs text-slate-500">
              No classifier model selected — only anomaly detection is active.
              Select a classifier model above to identify the arrhythmia type.
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}


// Interpretation Card — shows health information for the detected arrhythmia
function InterpretationCard({ info, onClose }: {
  info: any
  onClose: () => void
}) {
  const colorMap: Record<string, string> = {
    emerald: 'border-emerald-300 bg-emerald-50',
    amber: 'border-amber-300 bg-amber-50',
    rose: 'border-rose-300 bg-rose-50',
    slate: 'border-slate-300 bg-slate-50',
  }
  const bgClass = colorMap[info.color] || colorMap.slate

  return (
    <Card className={bgClass}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <Heart className="w-4 h-4 text-rose-500" />
            Interpretation: {info.name}
          </CardTitle>
          <Button size="sm" variant="ghost" onClick={onClose}>Close</Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {/* What is it */}
        <div>
          <h4 className="font-semibold text-slate-700 mb-1">What is it?</h4>
          <p className="text-slate-600 leading-relaxed">{info.what_is_it}</p>
        </div>

        {/* Causes */}
        {info.causes && info.causes.length > 0 && (
          <div>
            <h4 className="font-semibold text-slate-700 mb-1">What causes it?</h4>
            <ul className="list-disc list-inside text-slate-600 space-y-0.5">
              {info.causes.map((cause: string, i: number) => (
                <li key={i}>{cause}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Is it dangerous */}
        {info.is_dangerous && (
          <div>
            <h4 className="font-semibold text-slate-700 mb-1">Is it dangerous?</h4>
            <p className="text-slate-600 leading-relaxed">{info.is_dangerous}</p>
          </div>
        )}

        {/* When to see a doctor */}
        {info.when_to_see_doctor && info.when_to_see_doctor.length > 0 && (
          <div>
            <h4 className="font-semibold text-slate-700 mb-1">When to see a doctor</h4>
            <ul className="list-disc list-inside text-slate-600 space-y-0.5">
              {info.when_to_see_doctor.map((item: string, i: number) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Lifestyle tips */}
        {info.lifestyle_tips && info.lifestyle_tips.length > 0 && (
          <div>
            <h4 className="font-semibold text-slate-700 mb-1">Lifestyle tips</h4>
            <ul className="list-disc list-inside text-slate-600 space-y-0.5">
              {info.lifestyle_tips.map((tip: string, i: number) => (
                <li key={i}>{tip}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="text-xs text-slate-400 border-t pt-2 mt-2">
          This information is for educational purposes only and does not constitute medical advice.
          Always consult a qualified physician for proper diagnosis and treatment.
        </div>
      </CardContent>
    </Card>
  )
}
