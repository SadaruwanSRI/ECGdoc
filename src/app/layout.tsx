import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { AuthProvider } from "@/lib/auth-context";

export const metadata: Metadata = {
  title: "ECG Anomaly Detection — Clinical Decision Support",
  description:
    "Unsupervised deep-learning ECG anomaly detection system. Train autoencoders on MIT-BIH NSR DB, detect arrhythmias in real time, and generate physician-ready PDF reports.",
  keywords: ["ECG", "anomaly detection", "autoencoder", "deep learning", "MIT-BIH", "arrhythmia", "PyTorch", "FastAPI"],
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
