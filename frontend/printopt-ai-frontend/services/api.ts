import type {
  DatasetRecord,
  DatasetSummary,
  ModelStatus,
  ModelMetrics,
  PredictionResult,
  OptimizationRequest,
  OptimizationResult,
  ProcessMapData,
  SourceRecord,
  ApiResponse,
  ProcessParameters,
} from '../types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? '';

console.log('PRINTOPT API BASE:', API_BASE);

async function apiFetch<T>(
  path: string,
  options?: RequestInit
): Promise<ApiResponse<T>> {
  try {
    const requestUrl = `${API_BASE}${path}`;
    console.log('PRINTOPT REQUEST URL:', requestUrl);

    const res = await fetch(requestUrl, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });

    if (!res.ok) {
      return {
        status: 'error',
        error: `HTTP ${res.status}: ${res.statusText}`,
      };
    }

    const data = (await res.json()) as T;

    return {
      status: 'success',
      data,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Network error';

    return {
      status: 'error',
      error: message,
    };
  }
}

export const datasetApi = {
  getRecords: () => apiFetch<DatasetRecord[]>('/api/dataset'),

  getSummary: () =>
    apiFetch<DatasetSummary>('/api/dataset/summary'),
};

export const modelApi = {
  getStatus: () =>
    apiFetch<ModelStatus>('/api/model/status'),

  getMetrics: () =>
    apiFetch<ModelMetrics[]>('/api/model/metrics'),
};

interface FastApiPredictionResponse {
  input: {
    powder_size_um: number;
    laser_spot_um: number;
    laser_power_w: number;
    scanning_speed_mm_s: number;
    hatch_distance_um: number;
    layer_thickness_um: number;
    ved_j_mm3: number;
  };

  prediction: {
    uts_mpa: number;
    yield_strength_mpa: number;
    elongation_pct: number;
  };

  uncertainty: {
    uts_mpa: number;
    yield_strength_mpa: number;
    elongation_pct: number;
  };

  model: {
    name: string;
    material: string;
    process: string;
    features: string[];
  };
}

export const predictionApi = {
  predict: async (
    params: ProcessParameters
  ): Promise<ApiResponse<PredictionResult>> => {
    const response = await apiFetch<FastApiPredictionResponse>(
      '/predict',
      {
        method: 'POST',
        body: JSON.stringify({
          powder_size_um: params.powder_size_um,
          laser_spot_um: params.laser_spot_um,
          laser_power_w: params.laser_power_w,
          scanning_speed_mm_s: params.scan_speed_mm_s,
          hatch_distance_um: params.hatch_spacing_um,
          layer_thickness_um: params.layer_thickness_um,
        }),
      }
    );

    if (response.status === 'error' || !response.data) {
      return {
        status: 'error',
        error: response.error ?? 'Prediction failed',
      };
    }

    const apiResult = response.data;

    const result: PredictionResult = {
      uts_mpa: apiResult.prediction.uts_mpa,
      yield_strength_mpa:
        apiResult.prediction.yield_strength_mpa,
      elongation_pct:
        apiResult.prediction.elongation_pct,

      uts_uncertainty:
        apiResult.uncertainty.uts_mpa,
      yield_strength_uncertainty:
        apiResult.uncertainty.yield_strength_mpa,
      elongation_uncertainty:
        apiResult.uncertainty.elongation_pct,

      ved_j_mm3: apiResult.input.ved_j_mm3,

      model_version: apiResult.model.name,
    };

    return {
      status: 'success',
      data: result,
    };
  },
};

export const optimizationApi = {
  optimize: (request: OptimizationRequest) =>
    apiFetch<OptimizationResult>('/api/optimize', {
      method: 'POST',
      body: JSON.stringify(request),
    }),
};

export const processMapApi = {
  getData: () =>
    apiFetch<ProcessMapData>('/api/process-map'),
};

export const sourcesApi = {
  getSources: () =>
    apiFetch<SourceRecord[]>('/api/sources'),
};