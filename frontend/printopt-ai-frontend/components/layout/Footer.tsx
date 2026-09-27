import Link from 'next/link';
import {
  ArrowUpRight,
  ShieldCheck,
} from 'lucide-react';

const NAVIGATION = [
  { label: 'Overview', href: '/' },
  { label: 'Dataset', href: '/dataset' },
  { label: 'Prediction', href: '/prediction' },
  { label: 'Optimization', href: '/optimization' },
  { label: 'Process Map', href: '/process-map' },
  { label: 'Model', href: '/model' },
  { label: 'Sources', href: '/sources' },
];

export function Footer() {
  return (
    <footer className="mt-12 border-t border-[#1f2229]">
      <div className="max-w-[1280px] mx-auto px-6">

        {/* Main footer */}
        <div className="py-10 grid grid-cols-1 md:grid-cols-[1.4fr_1fr_1fr] gap-10">

          {/* Brand */}
          <div>
            <div className="flex items-center gap-2.5 mb-4">
              <div className="flex items-center justify-center w-7 h-7 rounded bg-[#76B900]/10 border border-[#76B900]/20">
                <span className="text-[11px] font-bold text-[#76B900]">
                  P
                </span>
              </div>

              <div>
                <div className="text-[13px] font-semibold tracking-tight text-[#e5e7eb]">
                  PrintOpt AI
                </div>

                <div className="text-[9px] uppercase tracking-[0.14em] text-[#4b505b]">
                  Research Platform
                </div>
              </div>
            </div>

            <p className="max-w-sm text-[12px] leading-relaxed text-[#626873]">
              AI-driven additive manufacturing process
              optimization for Laser Powder Bed Fusion
              of Ti-6Al-4V.
            </p>

            <div className="mt-5 flex items-center gap-2 text-[10px] uppercase tracking-wider text-[#4b505b]">
              <span className="h-1.5 w-1.5 rounded-full bg-[#76B900]" />
              LPBF · Ti-6Al-4V · Machine Learning
            </div>
          </div>

          {/* Project attribution */}
          <div>
            <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-[#4b505b] mb-4">
              Project Team
            </div>

            <div className="space-y-3">
              <div>
                <div className="text-[12px] text-[#c8cdd6]">
                  Hardik Sonu
                </div>
                <div className="text-[10px] text-[#50555f] mt-0.5">
                  Project Team
                </div>
              </div>

              <div>
                <div className="text-[12px] text-[#c8cdd6]">
                  Samreen Utta Ur Rehman
                </div>
                <div className="text-[10px] text-[#50555f] mt-0.5">
                  Project Team
                </div>
              </div>

              <div className="pt-1">
                <div className="text-[10px] uppercase tracking-wider text-[#4b505b] mb-1">
                  Supervisor
                </div>

                <div className="text-[12px] text-[#c8cdd6]">
                  Dr. Ing. Waseem Amin
                </div>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <div>
            <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-[#4b505b] mb-4">
              Platform
            </div>

            <div className="grid grid-cols-2 gap-y-2.5 gap-x-5">
              {NAVIGATION.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="group flex items-center gap-1 text-[11px] text-[#626873] hover:text-[#c8cdd6] transition-colors"
                >
                  {item.label}

                  <ArrowUpRight
                    size={9}
                    className="opacity-0 -translate-y-0.5 -translate-x-0.5 group-hover:opacity-100 group-hover:translate-x-0 transition-all"
                  />
                </Link>
              ))}

              <Link
                href="/privacy-policy"
                className="group flex items-center gap-1 text-[11px] text-[#626873] hover:text-[#c8cdd6] transition-colors"
              >
                Privacy Policy

                <ArrowUpRight
                  size={9}
                  className="opacity-0 -translate-y-0.5 -translate-x-0.5 group-hover:opacity-100 group-hover:translate-x-0 transition-all"
                />
              </Link>
            </div>
          </div>
        </div>

        {/* Institution */}
        <div className="border-t border-[#1f2229] py-5 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div>
            <div className="text-[11px] font-medium text-[#8b909a]">
              University of the Punjab, Lahore
            </div>

            <div className="text-[10px] text-[#4b505b] mt-1">
              Academic research and engineering project
            </div>
          </div>

          <div className="flex items-center gap-2 text-[10px] text-[#4b505b]">
            <ShieldCheck size={12} className="text-[#76B900]" />
            Research-focused platform
          </div>
        </div>

        {/* Copyright */}
        <div className="border-t border-[#1f2229] py-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <p className="text-[10px] text-[#454a54]">
            © 2026 PrintOpt AI. All rights reserved.
          </p>

          <p className="text-[10px] text-[#454a54]">
            Developed at the University of the Punjab, Lahore.
          </p>
        </div>
      </div>
    </footer>
  );
}