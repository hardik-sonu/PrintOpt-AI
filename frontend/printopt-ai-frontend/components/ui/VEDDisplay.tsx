'use client';

import { Zap } from 'lucide-react';
import { cn } from '@/lib/utils';

interface VEDDisplayProps {
  ved: number;
  className?: string;
}

export function VEDDisplay({ ved, className }: VEDDisplayProps) {
  const isOptimal = ved >= 50 && ved <= 120;
  const isTooHigh = ved > 120;
  const isTooLow = ved < 50;

  return (
    <div className={cn(
      'flex items-center gap-3 px-4 py-3 rounded border',
      isOptimal && 'border-[#76B900]/30 bg-[#76B900]/5',
      isTooHigh && 'border-[#f59e0b]/30 bg-[#f59e0b]/5',
      isTooLow && 'border-[#ef4444]/30 bg-[#ef4444]/5',
      className
    )}>
      <Zap
        size={16}
        className={cn(
          isOptimal && 'text-[#76B900]',
          isTooHigh && 'text-[#f59e0b]',
          isTooLow && 'text-[#ef4444]',
        )}
      />
      <div>
        <div className="text-[11px] text-[#5a5f6b] uppercase tracking-wider">Volumetric Energy Density</div>
        <div className="flex items-baseline gap-1.5 mt-0.5">
          <span className={cn(
            'text-xl font-semibold tabular-nums',
            isOptimal && 'text-[#76B900]',
            isTooHigh && 'text-[#fbbf24]',
            isTooLow && 'text-[#ef4444]',
          )}>
            {ved.toFixed(1)}
          </span>
          <span className="text-[12px] text-[#5a5f6b]">J/mm³</span>
        </div>
        <div className="text-[10px] text-[#3a3d47] mt-0.5">
          {isOptimal && 'Within typical LPBF process window'}
          {isTooHigh && 'High — risk of keyhole porosity'}
          {isTooLow && 'Low — risk of lack-of-fusion defects'}
        </div>
      </div>
    </div>
  );
}
