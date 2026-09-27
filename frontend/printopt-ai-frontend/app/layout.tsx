import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

import { Sidebar } from '@/components/layout/Sidebar';
import { Footer } from '@/components/layout/Footer';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: 'PrintOpt AI — LPBF Process Optimization',
  description:
    'AI-driven additive manufacturing process optimization for Laser Powder Bed Fusion of Ti-6Al-4V.',
  keywords: [
    'PrintOpt AI',
    'Additive Manufacturing',
    'LPBF',
    'Laser Powder Bed Fusion',
    'Ti-6Al-4V',
    'Machine Learning',
    'Process Optimization',
    'Materials Engineering',
  ],
  authors: [
    {
      name: 'Hardik Sonu',
    },
    {
      name: 'Samreen Utta Ur Rehman',
    },
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="bg-[#0f1012] text-[#f0f2f5] font-sans antialiased">
        <div className="min-h-screen flex">

          {/* Application sidebar */}
          <Sidebar />

          {/* Main application area */}
          <div className="flex-1 ml-56 min-h-screen flex flex-col">

            {/* Page content */}
            <main className="flex-1">
              <div className="max-w-[1280px] mx-auto px-6 py-6">
                {children}
              </div>
            </main>

            {/* Global footer */}
            <Footer />

          </div>
        </div>
      </body>
    </html>
  );
}