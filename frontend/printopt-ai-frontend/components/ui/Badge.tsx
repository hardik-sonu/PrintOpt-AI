import { cn } from '@/lib/utils';

type BadgeVariant = 'green' | 'amber' | 'red' | 'gray' | 'blue';

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  className?: string;
}

const VARIANTS: Record<BadgeVariant, string> = {
  green: 'border-[#76B900]/30 bg-[#76B900]/10 text-[#76B900]',
  amber: 'border-[#f59e0b]/30 bg-[#f59e0b]/10 text-[#fbbf24]',
  red: 'border-[#ef4444]/30 bg-[#ef4444]/10 text-[#f87171]',
  gray: 'border-[#2a2d35] bg-[#1a1d24] text-[#6b7280]',
  blue: 'border-[#3b82f6]/30 bg-[#3b82f6]/10 text-[#60a5fa]',
};

export function Badge({ children, variant = 'gray', className }: BadgeProps) {
  return (
    <span className={cn(
      'inline-flex items-center px-1.5 py-0.5 text-[10px] font-medium rounded border uppercase tracking-wider',
      VARIANTS[variant],
      className
    )}>
      {children}
    </span>
  );
}
