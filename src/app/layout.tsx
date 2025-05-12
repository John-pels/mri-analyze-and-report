import type React from "react"
import "@/app/globals.css"
import { Poppins } from "next/font/google"
// import { ThemeProvider } from "@/components/theme-provider"
// import { Toaster } from "@/components/ui/toaster"
// import { ThemeToggle } from "@/components/theme-toggle"

const poppins = Poppins({
  weight: ["400", "500", "600", "700"],
  subsets: ["latin"],
})

export const metadata = {
  title: "Medical Image Analysis Assistant",
  description: "Analyze medical images using CNN and LLM technology",
  icons: {
    icon: "/favicon.ico",
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={poppins.className}>
        {/* <ThemeProvider attribute="class" defaultTheme="system" enableSystem> */}
        <div className="fixed top-4 right-4 z-50">
          {/* <ThemeToggle /> */}
        </div>
        {children}
        {/* <Toaster />
        </ThemeProvider> */}
      </body>
    </html>
  )
}
