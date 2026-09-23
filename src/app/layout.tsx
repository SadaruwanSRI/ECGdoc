import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { AuthProvider } from "@/lib/auth-context";

export const metadata: Metadata = {
  title: "ECG Anomaly Detection - Research ECG Analysis",
  description:
    "Research ECG analysis system using a one-lead MLII temporal-holdout hierarchy with morphology, RR timing, and normal-autoencoder residual evidence.",
  keywords: [
    "ECG",
    "anomaly detection",
    "MLII",
    "temporal holdout",
    "autoencoder",
    "Extra Trees",
    "MIT-BIH",
    "arrhythmia",
    "FastAPI",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased bg-background text-foreground">
        <AuthProvider>{children}</AuthProvider>
        <Toaster />
      </body>
    </html>
  );
}
