// ─── Core Domain Types ───────────────────────────────────────────────────────

export interface DatasetRecord {
  record_id: string;
  laser_power_w: number;
  scan_speed_mm_s: number;
  layer_thickness_um: number;
  hatch_spacing_um: number;
  laser_spot_um?: number;
  powder_size_um?: number;
  ved_j_mm3: number; // always computed dynamically; never hardcoded
  uts_mpa?: number;
  yield_strength_mpa?: number;
  elongation_pct?: number;
  source: string;
}

export interface ProcessParameters {
  laser_power_w: number;
  scan_speed_mm_s: number;
  layer_thickness_um: number;
  hatch_spacing_um: number;
  laser_spot_um?: number;
  powder_size_um?: number;
}

export interface PredictionResult {
  uts_mpa?: number;
  yield_strength_mpa?: number;
  elongation_pct?: number;
  uts_uncertainty?: number;
  yield_strength_uncertainty?: number;
  elongation_uncertainty?: number;
  ved_j_mm3: number;
  coverage_warning?: string;
  model_version?: string;
  timestamp?: string;
}

export interface OptimizationObjective {
  target: 'maximize_uts' | 'maximize_yield_strength' | 'maximize_elongation' | 'multi_objective';
  weights?: {
    uts?: number;
    yield_strength?: number;
    elongation?: number;
  };
}

export interface ParameterConstraints {
  laser_power_w?: { min: number; max: number };
  scan_speed_mm_s?: { min: number; max: number };
  layer_thickness_um?: { min: number; max: number };
  hatch_spacing_um?: { min: number; max: number };
}

export interface OptimizationRequest {
  objective: OptimizationObjective;
  constraints: ParameterConstraints;
}

export interface OptimizationResult {
  recommended_parameters: ProcessParameters & { ved_j_mm3: number };
  predicted_properties: {
    uts_mpa?: number;
    yield_strength_mpa?: number;
    elongation_pct?: number;
  };
  confidence?: number;
  notes?: string;
}

export interface ModelMetrics {
  model_name: 'random_forest' | 'xgboost' | 'ann';
  target: 'uts' | 'yield_strength' | 'elongation';
  r2?: number;
  mae?: number;
  rmse?: number;
  n_samples?: number;
  trained_at?: string;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ModelStatus {
  is_connected: boolean;
  models_available: string[];
  last_trained?: string;
  dataset_size?: number;
}

export interface SourceRecord {
  id: string;
  title: string;
  authors?: string[];
  year?: number;
  doi?: string;
  source_type?: 'journal' | 'conference' | 'thesis' | 'report' | 'other';
  record_count: number;
  journal?: string;
}

export interface DatasetSummary {
  total_records: number;
  total_sources: number;
  process_parameters_count: number;
  prediction_targets_count: number;
  laser_power_range: [number, number];
  scan_speed_range: [number, number];
  layer_thickness_range: [number, number];
  hatch_spacing_range: [number, number];
  ved_range: [number, number];
  uts_range: [number, number];
  yield_strength_range: [number, number];
  elongation_range: [number, number];
}

export interface ProcessMapData {
  ved_values: number[];
  uts_values: (number | null)[];
  yield_strength_values: (number | null)[];
  elongation_values: (number | null)[];
  record_ids: string[];
  sources: string[];
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
  status: 'success' | 'error' | 'pending';
}

export type NavPage =
  | 'overview'
  | 'dataset'
  | 'prediction'
  | 'optimization'
  | 'process-map'
  | 'model'
  | 'sources';

export interface FilterState {
  search: string;
  source?: string;
  laser_power_range?: [number, number];
  scan_speed_range?: [number, number];
}
