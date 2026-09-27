import { cn } from '@/lib/utils';

interface CardProps {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
  headerRight?: React.ReactNode;
}

export function Card({ title, subtitle, children, className, headerRight }: CardProps) {
  return (
    <div className={cn('bg-[#16181c] border border-[#1f2229] rounded', className)}>
      {title && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#1f2229]">
          <div>
            <div className="text-[13px] font-semibold text-[#c8cdd6]">{title}</div>
            {subtitle && <div className="text-[11px] text-[#5a5f6b] mt-0.5">{subtitle}</div>}
          </div>
          {headerRight && <div>{headerRight}</div>}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}
