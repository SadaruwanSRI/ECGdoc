'use client'

import { useState } from 'react'
import { useAuth } from '@/lib/auth-context'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Activity, HeartPulse, AlertTriangle, LineChart } from 'lucide-react'

export function AuthScreen() {
  const { login, register } = useAuth()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Login form
  const [loginEmail, setLoginEmail] = useState('demo@ecg.local')
  const [loginPassword, setLoginPassword] = useState('demo1234')

  // Register form
  const [regEmail, setRegEmail] = useState('')
  const [regName, setRegName] = useState('')
  const [regPassword, setRegPassword] = useState('')

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      await login(loginEmail, loginPassword)
    } catch (err: any) {
      setError(err.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      await register(regEmail, regName, regPassword)
    } catch (err: any) {
      setError(err.message || 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen grid lg:grid-cols-2">
      {/* Left: hero panel */}
      <div className="hidden lg:flex flex-col justify-between p-12 bg-gradient-to-br from-rose-950 via-slate-950 to-slate-900 text-white relative overflow-hidden">
        <div className="absolute inset-0 opacity-10">
          <svg viewBox="0 0 1200 600" className="w-full h-full">
            <path
              d="M 0 300 L 200 300 L 220 280 L 240 320 L 260 200 L 280 400 L 300 300 L 500 300 L 520 280 L 540 320 L 560 200 L 580 400 L 600 300 L 800 300 L 820 280 L 840 320 L 860 200 L 880 400 L 900 300 L 1200 300"
              stroke="#fb7185" strokeWidth="2" fill="none"
            />
          </svg>
        </div>

        <div className="relative">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-12 h-12 rounded-xl bg-rose-500/20 border border-rose-400/30 flex items-center justify-center">
              <HeartPulse className="w-6 h-6 text-rose-400" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">ECG Anomaly Detection</h1>
              <p className="text-rose-300/70 text-sm">Research ECG Analysis System</p>
            </div>
          </div>
        </div>

        <div className="relative space-y-6">
          <h2 className="text-4xl font-bold leading-tight">
            One-lead MLII<br />
            <span className="text-rose-400">arrhythmia research</span>
          </h2>
          <p className="text-slate-300 text-lg leading-relaxed max-w-md">
            Combine MLII morphology, RR timing, and frozen normal-autoencoder
            residual evidence to detect and classify ECG anomalies in MIT-BIH
            records or research-grade live streams.
          </p>

          <div className="grid grid-cols-2 gap-4 max-w-md pt-4">
            <Feature icon={<Activity className="w-4 h-4" />} label="Real-time detection" />
            <Feature icon={<LineChart className="w-4 h-4" />} label="Model version comparison" />
            <Feature icon={<AlertTriangle className="w-4 h-4" />} label="Instant alerts" />
            <Feature icon={<HeartPulse className="w-4 h-4" />} label="Physician PDF reports" />
          </div>
        </div>

        <div className="relative text-xs text-slate-400">
          Research project — A.K.S.S. Wimalasena • University of Ruhuna • Supervisor: Ms. Sachini Karunarathne
        </div>
      </div>

      {/* Right: auth forms */}
      <div className="flex items-center justify-center p-8 bg-slate-50">
        <div className="w-full max-w-md">
          <div className="lg:hidden mb-8 text-center">
            <div className="w-14 h-14 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center mx-auto mb-3">
              <HeartPulse className="w-7 h-7 text-rose-500" />
            </div>
            <h1 className="text-2xl font-bold">ECG Anomaly Detection</h1>
            <p className="text-slate-500 text-sm">Research ECG Analysis</p>
          </div>

          <Tabs defaultValue="login">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="login">Sign In</TabsTrigger>
              <TabsTrigger value="register">Register</TabsTrigger>
            </TabsList>

            <TabsContent value="login">
              <Card>
                <CardHeader>
                  <CardTitle>Welcome back</CardTitle>
                  <CardDescription>
                    Sign in to access the ECG monitoring dashboard.
                  </CardDescription>
                </CardHeader>
                <form onSubmit={handleLogin}>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="email">Email</Label>
                      <Input
                        id="email" type="email" required
                        value={loginEmail}
                        onChange={(e) => setLoginEmail(e.target.value)}
                        placeholder="you@hospital.org"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="password">Password</Label>
                      <Input
                        id="password" type="password" required
                        value={loginPassword}
                        onChange={(e) => setLoginPassword(e.target.value)}
                      />
                    </div>
                    {error && (
                      <div className="text-sm text-rose-600 bg-rose-50 border border-rose-200 rounded-md p-3">
                        {error}
                      </div>
                    )}
                    <div className="text-xs text-slate-500 bg-slate-100 rounded p-2">
                      Demo: <code>demo@ecg.local</code> / <code>demo1234</code>
                    </div>
                  </CardContent>
                  <CardFooter>
                    <Button type="submit" disabled={loading} className="w-full">
                      {loading ? 'Signing in…' : 'Sign In'}
                    </Button>
                  </CardFooter>
                </form>
              </Card>
            </TabsContent>

            <TabsContent value="register">
              <Card>
                <CardHeader>
                  <CardTitle>Create an account</CardTitle>
                  <CardDescription>
                    Register as a clinician to start monitoring patients.
                  </CardDescription>
                </CardHeader>
                <form onSubmit={handleRegister}>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="r-name">Full name</Label>
                      <Input
                        id="r-name" required
                        value={regName}
                        onChange={(e) => setRegName(e.target.value)}
                        placeholder="Dr. Jane Smith"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="r-email">Email</Label>
                      <Input
                        id="r-email" type="email" required
                        value={regEmail}
                        onChange={(e) => setRegEmail(e.target.value)}
                        placeholder="you@hospital.org"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="r-password">Password</Label>
                      <Input
                        id="r-password" type="password" required minLength={6}
                        value={regPassword}
                        onChange={(e) => setRegPassword(e.target.value)}
                        placeholder="At least 6 characters"
                      />
                    </div>
                    {error && (
                      <div className="text-sm text-rose-600 bg-rose-50 border border-rose-200 rounded-md p-3">
                        {error}
                      </div>
                    )}
                  </CardContent>
                  <CardFooter>
                    <Button type="submit" disabled={loading} className="w-full">
                      {loading ? 'Creating account…' : 'Register'}
                    </Button>
                  </CardFooter>
                </form>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
}

function Feature({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-300">
      <div className="w-7 h-7 rounded-md bg-rose-500/10 border border-rose-400/20 flex items-center justify-center text-rose-400">
        {icon}
      </div>
      {label}
    </div>
  )
}
