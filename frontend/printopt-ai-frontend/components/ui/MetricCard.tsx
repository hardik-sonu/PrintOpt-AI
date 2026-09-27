import { cn } from '@/lib/utils';

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  sublabel?: string;
  accent?: boolean;
  className?: string;
}

export function MetricCard({ label, value, unit, sublabel, accent, className }: MetricCardProps) {
  return (
    <div className={cn(
      'bg-[#16181c] border border-[#1f2229] rounded px-4 py-3.5',
      accent && 'border-[#76B900]/25 bg-[#0d1a00]',
      className
    )}>
      <div className="text-[11px] font-medium text-[#5a5f6b] uppercase tracking-wider mb-2">{label}</div>
      <div className="flex items-baseline gap-1.5">
        <span className={cn('text-2xl font-semibold tracking-tight', accent ? 'text-[#76B900]' : 'text-[#f0f2f5]')}>
          {value}
        </span>
        {unit && <span className="text-[12px] text-[#5a5f6b]">{unit}</span>}
      </div>
      {sublabel && <div className="text-[11px] text-[#3a3d47] mt-1">{sublabel}</div>}
    </div>
  );
}
