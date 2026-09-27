import { AlertTriangle, Info, CheckCircle, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';

type BannerVariant = 'info' | 'warning' | 'success' | 'pending';

interface StatusBannerProps {
  variant: BannerVariant;
  title: string;
  message?: string;
  className?: string;
}

const ICONS = {
  info: Info,
  warning: AlertTriangle,
  success: CheckCircle,
  pending: Clock,
};

const STYLES = {
  info: 'border-[#3b82f6]/30 bg-[#3b82f6]/8 text-[#60a5fa]',
  warning: 'border-[#f59e0b]/30 bg-[#f59e0b]/8 text-[#fbbf24]',
  success: 'border-[#76B900]/30 bg-[#76B900]/8 text-[#76B900]',
  pending: 'border-[#5a5f6b]/30 bg-[#1a1d24] text-[#6b7280]',
};

export function StatusBanner({ variant, title, message, className }: StatusBannerProps) {
  const Icon = ICONS[variant];
  return (
    <div className={cn('flex items-start gap-3 px-4 py-3 rounded border text-[13px]', STYLES[variant], className)}>
      <Icon size={14} className="mt-0.5 flex-shrink-0" />
      <div>
        <span className="font-medium">{title}</span>
        {message && <span className="ml-1.5 opacity-80">{message}</span>}
      </div>
    </div>
  );
}
