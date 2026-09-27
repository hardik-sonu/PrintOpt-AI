'use client';

import { cn } from '@/lib/utils';

interface ParameterInputProps {
  label: string;
  unit: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  description?: string;
  className?: string;
}

export function ParameterInput({
  label,
  unit,
  value,
  min,
  max,
  step = 1,
  onChange,
  description,
  className,
}: ParameterInputProps) {
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className={cn('space-y-1.5', className)}>
      <div className="flex items-center justify-between">
        <label className="text-[12px] font-medium text-[#8b909a]">
          {label}
          {description && (
            <span className="ml-1 text-[#3a3d47]" title={description}>ⓘ</span>
          )}
        </label>
        <div className="flex items-center gap-1.5">
          <input
            type="number"
            value={value}
            min={min}
            max={max}
            step={step}
            onChange={(e) => onChange(parseFloat(e.target.value) || min)}
            className="w-20 text-right bg-[#0f1012] border border-[#2a2d35] rounded px-2 py-1 text-[12px] text-[#f0f2f5] focus:outline-none focus:border-[#76B900]/50 focus:ring-0"
          />
          <span className="text-[11px] text-[#5a5f6b] w-10">{unit}</span>
        </div>
      </div>
      <div className="relative h-1 bg-[#1f2229] rounded-full overflow-hidden">
        <div
          className="absolute left-0 top-0 h-full bg-[#76B900] rounded-full transition-all"
          style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-[#3a3d47]">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
