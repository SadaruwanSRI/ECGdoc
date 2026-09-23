'use client'

import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { FileText, Download, RefreshCw, History, AlertTriangle, Clock, Activity, Trash2 } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'

type Session = {
  id: string
  user_id: string
  model_id: string | null
  source_type: string
  source_detail: string | null
  started_at: string
  ended_at: string | null
  status: string
  total_beats: number
  anomaly_beats: number
  summary: any
}

type AlertItem = {
  id: string
  timestamp: string
  anomaly_score: number
  threshold: number
  severity: string
  message: string
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

export function SessionsTab() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState<Session | null>(null)
  const [alerts, setAlerts] = useState<AlertItem[]>([])
  const [reportDialog, setReportDialog] = useState(false)
  const [physicianName, setPhysicianName] = useState('')
  const [notes, setNotes] = useState('')
  const [generating, setGenerating] = useState(false)
  const [reportUrl, setReportUrl] = useState<string | null>(null)
  const { toast } = useToast()

  const refresh = async () => {
    setLoading(true)
    try {
      const r = await api.listSessions()
      setSessions(r.data?.sessions || [])
    } catch (e: any) {
      toast({ title: 'Error', description: e.message, variant: 'destructive' })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  const viewSession = async (s: Session) => {
    setSelected(s)
    try {
      const r = await api.getSession(s.id)
      setAlerts(r.data?.alerts || [])
    } catch (e: any) {
      toast({ title: 'Error', description: e.message, variant: 'destructive' })
    }
  }

  const handleDeleteSession = async (sessionId: string) => {
    if (!confirm('Delete this session and all its data (alerts, ECG data points)? This cannot be undone.')) return
    try {
      await api.deleteSession(sessionId)
      toast({ title: 'Session deleted' })
      if (selected?.id === sessionId) setSelected(null)
      refresh()
    } catch (e: any) {
      toast({ title: 'Delete failed', description: e.message, variant: 'destructive' })
    }
  }

  const generateReport = async () => {
    if (!selected) return
    setGenerating(true)
    try {
      const r = await api.generateReport(selected.id, physicianName || undefined, notes || undefined)
      if (r.ok && r.data?.filename) {
        setReportUrl(r.data.filename)
        toast({ title: 'Report generated', description: 'Click download to save the PDF.' })
      }
    } catch (e: any) {
      toast({ title: 'Report failed', description: e.message, variant: 'destructive' })
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="grid lg:grid-cols-4 gap-6">
      {/* Sessions list */}
      <div className="lg:col-span-3">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <History className="w-5 h-5 text-rose-500" />
                  Session History
                </CardTitle>
                <CardDescription>All recorded ECG monitoring sessions.</CardDescription>
              </div>
              <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
                <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {sessions.length === 0 ? (
              <div className="text-center py-12 text-slate-500">
                <Activity className="w-12 h-12 mx-auto mb-3 text-slate-300" />
                <p className="font-medium">No sessions yet</p>
                <p className="text-sm">Start a live analysis to record your first session.</p>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Started</TableHead>
                    <TableHead>Source</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Analyzed</TableHead>
                    <TableHead className="text-right">Flagged</TableHead>
                    <TableHead className="text-right">Anomaly %</TableHead>
                    <TableHead></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sessions.map(s => {
                    const pct = s.total_beats > 0 ? (s.anomaly_beats / s.total_beats) * 100 : 0
                    return (
                      <TableRow
                        key={s.id}
                        className={`cursor-pointer ${selected?.id === s.id ? 'bg-rose-50' : ''}`}
                        onClick={() => viewSession(s)}
                      >
                        <TableCell className="text-xs text-slate-600">
                          {new Date(s.started_at).toLocaleString()}
                        </TableCell>
                        <TableCell className="text-xs">
                          <span className="font-mono">{s.source_type}</span>
                          {s.source_detail && <span className="text-slate-400 ml-1">({s.source_detail})</span>}
                        </TableCell>
                        <TableCell>
                          <Badge variant={s.status === 'completed' ? 'default' : s.status === 'running' ? 'secondary' : 'outline'} className="text-xs">
                            {s.status}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs">{s.total_beats.toLocaleString()}</TableCell>
                        <TableCell className="text-right font-mono text-xs">
                          {s.anomaly_beats.toLocaleString()}
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs">
                          <span className={pct > 1 ? 'text-rose-600 font-semibold' : ''}>{pct.toFixed(2)}%</span>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-1">
                            {s.status === 'completed' && (
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  viewSession(s)
                                  setReportDialog(true)
                                }}
                              >
                                <FileText className="w-3 h-3 mr-1" />
                                Report
                              </Button>
                            )}
                            {s.status !== 'running' && (
                              <Button
                                size="sm"
                                variant="ghost"
                                className="text-rose-600 hover:text-rose-700 hover:bg-rose-50"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  handleDeleteSession(s.id)
                                }}
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </Button>
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Session detail */}
      <div className="lg:col-span-1 space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Session Detail</CardTitle>
          </CardHeader>
          <CardContent>
            {!selected ? (
              <p className="text-sm text-slate-500 text-center py-8">
                Click a session row to see details.
              </p>
            ) : (
              <div className="space-y-3 text-sm">
                <DetailRow label="Session ID" value={selected.id.slice(0, 12) + '…'} mono />
                <DetailRow label="Source" value={`${selected.source_type}${selected.source_detail ? ` (${selected.source_detail})` : ''}`} />
                <DetailRow label="Started" value={new Date(selected.started_at).toLocaleString()} />
                <DetailRow label="Ended" value={selected.ended_at ? new Date(selected.ended_at).toLocaleString() : '—'} />
                <DetailRow label="Status" value={selected.status} />
                <div className="border-t pt-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500 flex items-center gap-1.5"><Activity className="w-3 h-3" /> Total {selected.summary?.count_unit || 'beats'}</span>
                    <span className="font-mono font-semibold">{selected.total_beats.toLocaleString()}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500 flex items-center gap-1.5"><AlertTriangle className="w-3 h-3" /> Flagged {selected.summary?.count_unit || 'beats'}</span>
                    <span className={`font-mono font-semibold ${selected.anomaly_beats > 0 ? 'text-rose-600' : ''}`}>
                      {selected.anomaly_beats.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500 flex items-center gap-1.5"><Clock className="w-3 h-3" /> Duration</span>
                    <span className="font-mono">
                      {selected.summary?.duration_s ? `${selected.summary.duration_s.toFixed(1)}s` : '—'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-slate-500">Analysis mode</span>
                    <Badge variant="outline" className="text-[10px]">
                      {selected.summary?.analysis_mode === 'beat-aligned-hierarchical'
                        ? 'Beat + RR'
                        : 'Reconstruction'}
                    </Badge>
                  </div>
                </div>

                {selected.status === 'completed' && (
                  <Button
                    className="w-full mt-3"
                    onClick={() => setReportDialog(true)}
                  >
                    <FileText className="w-4 h-4 mr-2" />
                    Generate PDF Report
                  </Button>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Alerts for selected session */}
        {selected && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-rose-500" />
                Alerts ({alerts.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-80 w-full border rounded-md">
                {alerts.length === 0 ? (
                  <p className="text-sm text-slate-500 text-center py-6">No alerts for this session.</p>
                ) : (
                  <div className="divide-y">
                    {alerts.map(a => {
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
                        <div key={a.id} className={`p-2 ${a.severity === 'critical' ? 'bg-rose-50' : 'bg-amber-50'}`}>
                          <div className="flex items-start justify-between gap-2">
                            <div className="text-xs font-medium flex-1">{a.message}</div>
                            <Badge variant={a.severity === 'critical' ? 'destructive' : 'default'} className="text-xs shrink-0">
                              {a.severity}
                            </Badge>
                          </div>
                          <div className="text-xs text-slate-500 mt-0.5">
                            {ctx.score_type === 'abnormal_probability'
                              ? `abnormal probability=${(a.anomaly_score * 100).toFixed(1)}%`
                              : `score=${a.anomaly_score.toFixed(4)}`}
                            {' • '}threshold={a.threshold.toFixed(4)}{' • '}
                            {new Date(a.timestamp).toLocaleTimeString()}
                          </div>
                          {hasSignal && (
                            <div className="mt-1 h-16 bg-white rounded border border-slate-200 overflow-hidden">
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
                                  <Line type="monotone" dataKey="value" stroke="#dc2626" strokeWidth={1.1} dot={false} isAnimationActive={false} />
                                  <Line type="monotone" dataKey="recon" stroke="#0891b2" strokeWidth={0.9} strokeDasharray="3 2" dot={false} isAnimationActive={false} connectNulls />
                                </LineChart>
                              </ResponsiveContainer>
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
        )}
      </div>

      {/* Report generation dialog */}
      <Dialog open={reportDialog} onOpenChange={setReportDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Generate Physician Report</DialogTitle>
            <DialogDescription>
              Create a PDF report for physician review. Includes ECG waveform, alerts table, and clinical summary.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="physician">Physician name (optional)</Label>
              <Input
                id="physician"
                value={physicianName}
                onChange={(e) => setPhysicianName(e.target.value)}
                placeholder="Dr. Jane Smith"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="notes">Notes (optional)</Label>
              <Textarea
                id="notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Clinical context, symptoms observed, follow-up plan…"
                rows={4}
              />
            </div>
            {reportUrl && (
              <div className="bg-emerald-50 border border-emerald-200 rounded-md p-3 text-sm">
                <div className="font-medium text-emerald-700 mb-2">Report ready!</div>
                <button
                  type="button"
                  onClick={() => {
                    api.downloadReport(reportUrl).catch((e: Error) => {
                      toast({ title: 'Download failed', description: e.message, variant: 'destructive' })
                    })
                  }}
                  className="inline-flex items-center gap-2 text-emerald-700 hover:underline"
                >
                  <Download className="w-4 h-4" />
                  Download PDF
                </button>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => { setReportDialog(false); setReportUrl(null); }}>
              Close
            </Button>
            <Button onClick={generateReport} disabled={generating}>
              {generating ? 'Generating…' : reportUrl ? 'Regenerate' : 'Generate Report'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

function DetailRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-slate-500 text-xs">{label}</span>
      <span className={`text-sm ${mono ? 'font-mono' : ''}`}>{value}</span>
    </div>
  )
}
