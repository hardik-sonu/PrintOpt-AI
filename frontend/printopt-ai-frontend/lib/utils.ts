export function cn(...classes: (string | undefined | false | null)[]): string {
  return classes.filter(Boolean).join(' ');
}

export function formatNumber(n: number | undefined | null, decimals = 1): string {
  if (n == null || isNaN(n)) return '—';
  return n.toFixed(decimals);
}

export function formatRange(min: number, max: number, unit = ''): string {
  return `${min.toFixed(0)}–${max.toFixed(0)}${unit ? ' ' + unit : ''}`;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
