import type { Metadata } from "next";
import { Toaster } from "sonner";
import "./globals.css";

export const metadata: Metadata = {
  title: "Deep Research Agent",
  description: "Multi-provider AI deep research agent with glassmorphic UI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="relative">
        {/* Animated mesh gradient background */}
        <div className="bg-mesh" />

        {/* Content layer */}
        <div className="relative z-10">
          {children}
        </div>
        <Toaster
          position="top-right"
          richColors
          toastOptions={{
            style: {
              background: "rgba(255,255,255,0.06)",
              backdropFilter: "blur(20px)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: "#f0f0f5",
            },
          }}
        />
      </body>
    </html>
  );
}
