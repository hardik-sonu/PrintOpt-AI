/** Default parameter ranges based on Ti-6Al-4V LPBF experimental literature */
export const PARAMETER_RANGES = {
  laser_power_w: { min: 50, max: 400, step: 5, default: 200 },
  scan_speed_mm_s: { min: 200, max: 1800, step: 10, default: 1000 },
  layer_thickness_um: { min: 20, max: 80, step: 5, default: 30 },
  hatch_spacing_um: { min: 60, max: 180, step: 5, default: 120 },
  laser_spot_um: { min: 50, max: 150, step: 5, default: 80 },
  powder_size_um: { min: 10, max: 63, step: 1, default: 30 },
} as const;

export const DATASET_STATS = {
  total_records: 173,
  total_sources: 34,
  process_parameters_count: 4,
  prediction_targets_count: 3,
} as const;

export const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', href: '/' },
  { id: 'dataset', label: 'Dataset', href: '/dataset' },
  { id: 'prediction', label: 'Prediction', href: '/prediction' },
  { id: 'optimization', label: 'Optimization', href: '/optimization' },
  { id: 'process-map', label: 'Process Map', href: '/process-map' },
  { id: 'model', label: 'Model', href: '/model' },
  { id: 'sources', label: 'Sources', href: '/sources' },
] as const;

export const COLORS = {
  green: '#76B900',
  greenDim: '#5a8c00',
  greenMuted: '#3d6000',
  bg: '#0f1012',
  surface: '#16181c',
  surfaceElevated: '#1d2026',
  border: '#2a2d35',
  borderStrong: '#3a3d47',
  textPrimary: '#f0f2f5',
  textSecondary: '#8b909a',
  textMuted: '#5a5f6b',
  amber: '#f59e0b',
  red: '#ef4444',
  blue: '#3b82f6',
} as const;
