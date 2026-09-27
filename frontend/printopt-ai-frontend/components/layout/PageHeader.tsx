import { cn } from '@/lib/utils';

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  badge?: string;
  actions?: React.ReactNode;
  className?: string;
}

export function PageHeader({ title, subtitle, badge, actions, className }: PageHeaderProps) {
  return (
    <div className={cn('flex items-start justify-between mb-6', className)}>
      <div>
        <div className="flex items-center gap-2.5 mb-1">
          <h1 className="text-lg font-semibold text-[#f0f2f5] tracking-tight">{title}</h1>
          {badge && (
            <span className="px-2 py-0.5 text-[10px] font-medium rounded border border-[#76B900]/30 text-[#76B900] bg-[#76B900]/8 uppercase tracking-wider">
              {badge}
            </span>
          )}
        </div>
        {subtitle && (
          <p className="text-[13px] text-[#5a5f6b]">{subtitle}</p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}
