'use client'

import { useState } from 'react'
import { useAuth } from '@/lib/auth-context'
import { AuthScreen } from '@/components/auth-screen'
import { TrainingTab } from '@/components/training-tab'
import { DashboardTab } from '@/components/dashboard-tab'
import { LiveTab } from '@/components/live-tab'
import { SessionsTab } from '@/components/sessions-tab'
import { PerformanceTab } from '@/components/performance-tab'
import { Button } from '@/components/ui/button'
import { HeartPulse, Activity, GitCompare, Radio, History, LogOut, Brain, BarChart3 } from 'lucide-react'
import {
  Tabs, TabsContent, TabsList, TabsTrigger,
} from '@/components/ui/tabs'

export default function Home() {
  const { user, loading, logout } = useAuth()
  const [tab, setTab] = useState('live')

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="text-center">
          <div className="w-12 h-12 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center mx-auto mb-3 animate-pulse">
            <HeartPulse className="w-6 h-6 text-rose-500" />
          </div>
          <p className="text-slate-500">Loading…</p>
        </div>
      </div>
    )
  }

  if (!user) {
    return <AuthScreen />
  }

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      {/* Header */}
      <header className="border-b bg-white sticky top-0 z-10">
        <div className="w-full px-4 lg:px-8 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-rose-500/10 border border-rose-500/20 flex items-center justify-center">
              <HeartPulse className="w-5 h-5 text-rose-500" />
            </div>
            <div>
              <h1 className="font-semibold text-lg leading-tight">ECG Anomaly Detection</h1>
              <p className="text-xs text-slate-500 leading-tight">Clinical Decision Support</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 text-sm">
              <div className="w-8 h-8 rounded-full bg-rose-100 flex items-center justify-center text-rose-700 font-medium">
                {user.name?.[0] || user.email[0].toUpperCase()}
              </div>
              <div className="text-right">
                <div className="font-medium leading-tight">{user.name || user.email}</div>
                <div className="text-xs text-slate-500 leading-tight">{user.role}</div>
              </div>
            </div>
            <Button variant="ghost" size="sm" onClick={logout}>
              <LogOut className="w-4 h-4 mr-2" />
              Sign Out
            </Button>
          </div>
        </div>
      </header>

      {/* Main content — full width */}
      <main className="flex-1 w-full px-4 lg:px-8 py-6">
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList className="grid w-full grid-cols-2 sm:grid-cols-5 mb-6">
            <TabsTrigger value="live" className="flex items-center gap-2">
              <Radio className="w-4 h-4" />
              <span className="hidden sm:inline">Live Analysis</span>
              <span className="sm:hidden">Live</span>
            </TabsTrigger>
            <TabsTrigger value="training" className="flex items-center gap-2">
              <Brain className="w-4 h-4" />
              <span className="hidden sm:inline">Train Model</span>
              <span className="sm:hidden">Train</span>
            </TabsTrigger>
            <TabsTrigger value="performance" className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              <span className="hidden sm:inline">Performance</span>
              <span className="sm:hidden">Test</span>
            </TabsTrigger>
            <TabsTrigger value="dashboard" className="flex items-center gap-2">
              <GitCompare className="w-4 h-4" />
              <span className="hidden sm:inline">Compare Models</span>
              <span className="sm:hidden">Compare</span>
            </TabsTrigger>
            <TabsTrigger value="sessions" className="flex items-center gap-2">
              <History className="w-4 h-4" />
              <span className="hidden sm:inline">Sessions</span>
              <span className="sm:hidden">Sessions</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="live"><LiveTab /></TabsContent>
          <TabsContent value="training"><TrainingTab /></TabsContent>
          <TabsContent value="performance"><PerformanceTab /></TabsContent>
          <TabsContent value="dashboard"><DashboardTab /></TabsContent>
          <TabsContent value="sessions"><SessionsTab /></TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t bg-white">
        <div className="w-full px-4 lg:px-8 py-4 text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Activity className="w-3 h-3" />
            ECG Anomaly Detection System v1.0
          </div>
          <div className="text-center sm:text-right">
            A.K.S.S. Wimalasena • University of Ruhuna • Supervisor: Ms. Sachini Karunarathne
          </div>
        </div>
      </footer>
    </div>
  )
}
