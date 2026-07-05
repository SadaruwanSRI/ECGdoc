'use client'

import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, BarChart, Bar } from 'recharts'
import { GitCompare, TrendingDown, Award, Layers, Activity, Cpu, Trash2 } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'

type ModelRow = {
  id: string
  name: string
  version: string
  architecture: string
  parameters: number
  status: string
  threshold: number | null
  created_at: string
  metrics: Record<string, { epoch: number; value: number }[]>
}

export function DashboardTab() {
  const [models, setModels] = useState<any[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [comparison, setComparison] = useState<{
    models: ModelRow[]
    metric_names: string[]
  } | null>(null)
  const [metricView, setMetricView] = useState<'curves' | 'bars'>('curves')
  const { toast } = useToast()

  useEffect(() => {
    refreshModels()
  }, [])

  const handleDeleteModel = async (modelId: string, modelName: string) => {
    if (!confirm(`Delete model "${modelName}"? This permanently removes the model file and all its metrics. This cannot be undone.`)) return
    try {
      await api.deleteModel(modelId)
      toast({ title: 'Model deleted', description: modelName })
      // Remove from selection if selected
      const next = new Set(selected)
      next.delete(modelId)
      setSelected(next)
      refreshModels()
    } catch (e: any) {
      toast({ title: 'Delete failed', description: e.message, variant: 'destructive' })
    }
  }

  const refreshModels = async () => {
    try {
      const r = await api.listModels()
      const list = r.data?.models || []
      setModels(list)
      // Auto-select the 2 most recent ready models
      const ready = list.filter((m: any) => m.status === 'ready').slice(0, 3)
      setSelected(new Set(ready.map((m: any) => m.id)))
    } catch (e) {
      console.error(e)
    }
  }

  // Auto-load comparison when selection changes
  useEffect(() => {
    if (selected.size < 2) {
      setComparison(null)
      return
    }
    api.compareModels(Array.from(selected))
      .then(r => setComparison(r.data))
      .catch(console.error)
  }, [selected])

  const toggle = (id: string) => {
    const next = new Set(selected)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    setSelected(next)
  }

  // Build chart data: per-epoch loss curves for each selected model
  const lossCurveData: any[] = []
  if (comparison) {
    // Find max epoch count
    const maxEpoch = Math.max(...comparison.models.flatMap(m =>
      (m.metrics.val_loss || []).map((x: any) => x.epoch)
    ), 0)
    for (let e = 1; e <= maxEpoch; e++) {
      const row: any = { epoch: e }
      comparison.models.forEach(m => {
        const vl = (m.metrics.val_loss || []).find((x: any) => x.epoch === e)
        const tl = (m.metrics.train_loss || []).find((x: any) => x.epoch === e)
        row[`${m.name}_train`] = tl?.value ?? null
        row[`${m.name}_val`] = vl?.value ?? null
      })
      lossCurveData.push(row)
    }
  }

  // Bar chart data: final metrics per model
  const barData = comparison?.models.map(m => ({
    name: m.name.length > 16 ? m.name.slice(0, 14) + '…' : m.name,
    val_loss: m.metrics.final_val_loss?.[0]?.value ?? null,
    threshold: m.metrics.threshold?.[0]?.value ?? null,
  })) || []

  const COLORS = ['#e11d48', '#0891b2', '#7c3aed', '#ea580c', '#16a34a', '#ca8a04']

  return (
    <div className="space-y-6">
      <div className="grid lg:grid-cols-4 gap-6">
        {/* Model selector */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <GitCompare className="w-5 h-5 text-rose-500" />
                Select Models
              </CardTitle>
              <CardDescription>
                Pick 2+ models to compare. Selected: {selected.size}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                <div className="space-y-2">
                  {models.length === 0 && (
                    <p className="text-sm text-slate-500 text-center py-8">
                      No trained models yet. Train one in the Training tab first.
                    </p>
                  )}
                  {models.map((m: any) => (
                    <label
                      key={m.id}
                      className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                        selected.has(m.id)
                          ? 'bg-rose-50 border-rose-300'
                          : 'bg-white border-slate-200 hover:bg-slate-50'
                      }`}
                    >
                      <Checkbox
                        checked={selected.has(m.id)}
                        onCheckedChange={() => toggle(m.id)}
                        className="mt-1"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm truncate">{m.name}</div>
                        <div className="text-xs text-slate-500 font-mono">{m.version}</div>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge
                            variant={m.status === 'ready' ? 'default' : m.status === 'failed' ? 'destructive' : 'outline'}
                            className="text-xs"
                          >
                            {m.status}
                          </Badge>
                          <span className="text-xs text-slate-500">{new Date(m.created_at).toLocaleString()}</span>
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        {/* Comparison panel */}
        <div className="lg:col-span-3 space-y-4">
          {!comparison || comparison.models.length < 2 ? (
            <Card className="border-dashed">
              <CardContent className="py-16 text-center text-slate-500">
                <GitCompare className="w-12 h-12 mx-auto mb-4 text-slate-300" />
                <p className="text-lg font-medium">Select at least 2 models to compare</p>
                <p className="text-sm">Choose models from the list on the left to view side-by-side metrics.</p>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Summary table */}
              <Card>
                <CardHeader>
                  <CardTitle>Side-by-Side Comparison</CardTitle>
                  <CardDescription>{comparison.models.length} models selected</CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Property</TableHead>
                        {comparison.models.map(m => (
                          <TableHead key={m.id} className="font-semibold">
                            {m.name}
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableRow>
                        <TableCell className="text-slate-500">Version</TableCell>
                        {comparison.models.map(m => <TableCell key={m.id} className="font-mono text-xs">{m.version}</TableCell>)}
                      </TableRow>
                      <TableRow>
                        <TableCell className="text-slate-500">Architecture</TableCell>
                        {comparison.models.map(m => <TableCell key={m.id} className="text-xs">{m.architecture}</TableCell>)}
                      </TableRow>
                      <TableRow>
                        <TableCell className="text-slate-500">Parameters</TableCell>
                        {comparison.models.map(m => <TableCell key={m.id} className="font-mono text-xs">{(m.parameters / 1e6).toFixed(2)}M</TableCell>)}
                      </TableRow>
                      <TableRow>
                        <TableCell className="text-slate-500">Threshold (MAE)</TableCell>
                        {comparison.models.map(m => <TableCell key={m.id} className="font-mono text-xs">{m.threshold?.toFixed(4) || '—'}</TableCell>)}
                      </TableRow>
                      <TableRow>
                        <TableCell className="text-slate-500">Created</TableCell>
                        {comparison.models.map(m => <TableCell key={m.id} className="text-xs">{new Date(m.created_at).toLocaleString()}</TableCell>)}
                      </TableRow>
                      <TableRow>
                        <TableCell className="text-slate-500">Status</TableCell>
                        {comparison.models.map(m => (
                          <TableCell key={m.id}>
                            <Badge variant={m.status === 'ready' ? 'default' : 'destructive'} className="text-xs">{m.status}</Badge>
                          </TableCell>
                        ))}
                      </TableRow>
                      <TableRow>
                        <TableCell className="text-slate-500">Final val loss</TableCell>
                        {comparison.models.map(m => (
                          <TableCell key={m.id} className="font-mono text-xs">
                            {m.metrics.final_val_loss?.[0]?.value?.toFixed(6) || '—'}
                          </TableCell>
                        ))}
                      </TableRow>
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              {/* Chart view toggle */}
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <TrendingDown className="w-5 h-5 text-rose-500" />
                  Metrics Visualization
                </h3>
                <Select value={metricView} onValueChange={(v: any) => setMetricView(v)}>
                  <SelectTrigger className="w-40"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="curves">Loss curves</SelectItem>
                    <SelectItem value="bars">Bar comparison</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {metricView === 'curves' && lossCurveData.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Validation Loss by Epoch</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={320}>
                      <LineChart data={lossCurveData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} />
                        <YAxis stroke="#64748b" fontSize={12} />
                        <Tooltip
                          contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12 }}
                          formatter={(v: number) => v?.toFixed(6) || '—'}
                        />
                        <Legend wrapperStyle={{ fontSize: 12 }} />
                        {comparison.models.map((m, i) => (
                          <Line
                            key={m.id}
                            type="monotone"
                            dataKey={`${m.name}_val`}
                            name={`${m.name} (val)`}
                            stroke={COLORS[i % COLORS.length]}
                            strokeWidth={2}
                            dot={{ r: 3 }}
                            connectNulls
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}

              {metricView === 'bars' && barData.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Final Metrics Comparison</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={320}>
                      <BarChart data={barData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
                        <YAxis stroke="#64748b" fontSize={12} />
                        <Tooltip
                          contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12 }}
                          formatter={(v: number) => v?.toFixed(6) || '—'}
                        />
                        <Legend wrapperStyle={{ fontSize: 12 }} />
                        <Bar dataKey="val_loss" name="Final val loss" fill="#e11d48" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="threshold" name="Threshold" fill="#0891b2" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}

              {/* Best model highlight */}
              {comparison.models.length > 0 && (
                <Card className="bg-gradient-to-br from-rose-50 to-amber-50 border-rose-200">
                  <CardContent className="pt-6">
                    <div className="flex items-center gap-3">
                      <Award className="w-8 h-8 text-amber-500" />
                      <div>
                        <div className="text-xs text-slate-500 uppercase tracking-wide">Best validation loss</div>
                        <div className="font-bold text-lg">
                          {(() => {
                            const sorted = [...comparison.models].sort((a, b) => {
                              const aV = a.metrics.final_val_loss?.[0]?.value ?? Infinity
                              const bV = b.metrics.final_val_loss?.[0]?.value ?? Infinity
                              return aV - bV
                            })
                            const best = sorted[0]
                            return `${best.name} (${best.metrics.final_val_loss?.[0]?.value?.toFixed(6) ?? '—'})`
                          })()}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </div>
      </div>

      {/* Model cards (all models, not just selected) */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-rose-500" />
            All Model Versions
          </CardTitle>
          <CardDescription>Every model trained in this system.</CardDescription>
        </CardHeader>
        <CardContent>
          {models.length === 0 ? (
            <p className="text-sm text-slate-500 text-center py-8">No models yet.</p>
          ) : (
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {models.map((m: any) => (
                <div key={m.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow relative group">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold text-sm truncate">{m.name}</div>
                      <div className="text-xs text-slate-500 font-mono">{m.version}</div>
                    </div>
                    <Badge variant={m.status === 'ready' ? 'default' : m.status === 'failed' ? 'destructive' : 'outline'} className="text-xs ml-2">
                      {m.status}
                    </Badge>
                  </div>
                  <div className="space-y-1 text-xs text-slate-600">
                    <div className="flex items-center gap-2"><Cpu className="w-3 h-3" /> {(m.parameters / 1e6).toFixed(2)}M params</div>
                    <div className="flex items-center gap-2"><Activity className="w-3 h-3" /> Threshold: {m.threshold?.toFixed(4) || '—'}</div>
                    <div className="text-slate-400">{new Date(m.created_at).toLocaleString()}</div>
                  </div>
                  {/* Delete button - appears on hover */}
                  <Button
                    size="sm"
                    variant="ghost"
                    className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity h-7 w-7 p-0 text-rose-600 hover:text-rose-700 hover:bg-rose-50"
                    onClick={() => handleDeleteModel(m.id, m.name)}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
