'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Database,
  BrainCircuit,
  Target,
  GitBranch,
  BarChart2,
  BookOpen,
  Cpu,
} from 'lucide-react';
import { cn } from '@/lib/utils';

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', href: '/', icon: LayoutDashboard },
  { id: 'dataset', label: 'Dataset', href: '/dataset', icon: Database },
  { id: 'prediction', label: 'Prediction', href: '/prediction', icon: BrainCircuit },
  { id: 'optimization', label: 'Optimization', href: '/optimization', icon: Target },
  { id: 'process-map', label: 'Process Map', href: '/process-map', icon: GitBranch },
  { id: 'model', label: 'Model', href: '/model', icon: BarChart2 },
  { id: 'sources', label: 'Sources', href: '/sources', icon: BookOpen },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-56 bg-[#12141a] border-r border-[#1f2229] flex flex-col z-40">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-[#1f2229]">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 bg-[#76B900] rounded flex items-center justify-center flex-shrink-0">
            <Cpu size={14} className="text-black" />
          </div>
          <div>
            <div className="text-[13px] font-semibold text-[#f0f2f5] tracking-wide">PrintOpt AI</div>
            <div className="text-[10px] text-[#5a5f6b] tracking-wider uppercase">LPBF · Ti-6Al-4V</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-3 px-2 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map(({ id, label, href, icon: Icon }) => {
          const isActive = pathname === href || (href !== '/' && pathname.startsWith(href));
          return (
            <Link
              key={id}
              href={href}
              className={cn(
                'flex items-center gap-2.5 px-3 py-2 rounded text-[13px] font-medium transition-colors',
                isActive
                  ? 'bg-[#1a2a00] text-[#76B900]'
                  : 'text-[#6b7280] hover:text-[#c8cdd6] hover:bg-[#1a1d24]'
              )}
            >
              <Icon size={15} className={isActive ? 'text-[#76B900]' : 'text-current'} />
              {label}
              {isActive && (
                <span className="ml-auto w-1 h-1 rounded-full bg-[#76B900]" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-[#1f2229]">
        <div className="text-[10px] text-[#3a3d47] leading-relaxed">
          <div>Ti-6Al-4V · LPBF</div>
          <div>ML Process Optimization</div>
          <div className="mt-1 text-[#2a2d35]" aria-hidden>v0.1.0-alpha</div>
        </div>
      </div>
    </aside>
  );
}
