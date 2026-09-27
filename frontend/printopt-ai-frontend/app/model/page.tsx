'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  RefreshCw,
} from 'lucide-react';

import { PageHeader } from '@/components/layout/PageHeader';
import { Card } from '@/components/ui/Card';
import { StatusBanner } from '@/components/layout/StatusBanner';
import { Badge } from '@/components/ui/Badge';

interface ModelMetric {
  model_name: string;
  target: string;
  r2: number | null;
  mae: number | null;
  rmse: number | null;
}

interface FeatureImportance {
  model_name: string;
  target: string;
  feature: string;
  importance: number | null;
}

interface PredictionRecord {
  experiment_id: string;
  source_id: string;

  uts_actual: number | null;
  ys_actual: number | null;
  elongation_actual: number | null;

  rf_uts_pred: number | null;
  rf_ys_pred: number | null;
  rf_elongation_pred: number | null;
}

interface ModelStatus {
  is_connected: boolean;
  models_available: string[];
  active_model: string;
  model_name: string;
  material: string;
  process: string;
  dataset_size: number;
  evaluation_samples: number;
  evaluation_type: string;
}

interface DiagnosticsResponse {
  feature_importance: FeatureImportance[];
  predictions: PredictionRecord[];
}

const MODEL_ORDER = [
  {
    id: 'Random Forest V2',
    label: 'Random Forest',
    abbr: 'RF',
  },
  {
    id: 'XGBoost V2',
    label: 'XGBoost',
    abbr: 'XGB',
  },
  {
    id: 'MLP V2',
    label: 'Multilayer Perceptron',
    abbr: 'MLP',
  },
];

const TARGETS = [
  {
    id: 'UTS_MPa',
    label: 'UTS',
    unit: 'MPa',
  },
  {
    id: 'YS_MPa',
    label: 'Yield Strength',
    unit: 'MPa',
  },
  {
    id: 'Elongation_pct',
    label: 'Elongation',
    unit: '%',
  },
];

const FEATURE_LABELS: Record<string, string> = {
  Powder_Size_um: 'Powder Size',
  Laser_Spot_um: 'Laser Spot',
  Laser_Power_W: 'Laser Power',
  Scanning_Speed_mm_s: 'Scan Speed',
  Hatch_Distance_um: 'Hatch Spacing',
  Layer_Thickness_um: 'Layer Thickness',
  VED_J_mm3: 'VED',
};

function formatMetric(value: number | null, digits = 3) {
  if (value === null || !Number.isFinite(value)) {
    return '—';
  }

  return value.toFixed(digits);
}

function getMetric(
  metrics: ModelMetric[],
  modelName: string,
  target: string
) {
  return metrics.find(
    (item) =>
      item.model_name === modelName &&
      item.target === target
  );
}

function getFeatureImportance(
  importance: FeatureImportance[],
  target: string
) {
  return importance
    .filter(
      (item) =>
        item.model_name === 'Random Forest V2' &&
        item.target === target &&
        item.importance !== null
    )
    .sort(
      (a, b) =>
        (b.importance ?? 0) -
        (a.importance ?? 0)
    );
}

export default function ModelPage() {
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [diagnostics, setDiagnostics] =
    useState<DiagnosticsResponse | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadModelData() {
  setLoading(true);
  setError(null);

  const API_BASE =
    process.env.NEXT_PUBLIC_API_URL ??
    'http://127.0.0.1:8000';

  try {
    const [
      statusResponse,
      metricsResponse,
      diagnosticsResponse,
    ] = await Promise.all([
      fetch(`${API_BASE}/api/model/status`),
      fetch(`${API_BASE}/api/model/metrics`),
      fetch(`${API_BASE}/api/model/diagnostics`),
    ]);

    if (
      !statusResponse.ok ||
      !metricsResponse.ok ||
      !diagnosticsResponse.ok
    ) {
      throw new Error(
        'Unable to load model evaluation data.'
      );
    }

    const statusData =
      (await statusResponse.json()) as ModelStatus;

    const metricsData =
      (await metricsResponse.json()) as ModelMetric[];

    const diagnosticsData =
      (await diagnosticsResponse.json()) as DiagnosticsResponse;

    setStatus(statusData);
    setMetrics(metricsData);
    setDiagnostics(diagnosticsData);
  } catch (err) {
    setError(
      err instanceof Error
        ? err.message
        : 'Unable to load model data.'
    );
  } finally {
    setLoading(false);
  }
}

  useEffect(() => {
    loadModelData();
  }, []);

  const rfFeatureImportance = useMemo(() => {
    if (!diagnostics) return [];

    const grouped = new Map<
      string,
      number
    >();

    diagnostics.feature_importance
      .filter(
        (item) =>
          item.model_name === 'Random Forest V2' &&
          item.importance !== null
      )
      .forEach((item) => {
        const current = grouped.get(item.feature) ?? 0;
        grouped.set(
          item.feature,
          current + (item.importance ?? 0)
        );
      });

    return Array.from(grouped.entries())
      .map(([feature, importance]) => ({
        feature,
        importance: importance / 3,
      }))
      .sort((a, b) => b.importance - a.importance);
  }, [diagnostics]);

  const parityData = useMemo(() => {
    if (!diagnostics) return [];

    return diagnostics.predictions
      .filter(
        (row) =>
          row.uts_actual !== null &&
          row.rf_uts_pred !== null
      )
      .map((row) => ({
        actual: row.uts_actual as number,
        predicted: row.rf_uts_pred as number,
      }));
  }, [diagnostics]);

  const residualData = useMemo(() => {
    if (!diagnostics) return [];

    return diagnostics.predictions
      .filter(
        (row) =>
          row.uts_actual !== null &&
          row.rf_uts_pred !== null
      )
      .map((row) => ({
        actual: row.uts_actual as number,
        residual:
          (row.rf_uts_pred as number) -
          (row.uts_actual as number),
      }));
  }, [diagnostics]);

  const maxFeatureImportance =
    rfFeatureImportance.length > 0
      ? Math.max(
          ...rfFeatureImportance.map(
            (item) => item.importance
          )
        )
      : 1;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Model"
        subtitle="ML model evaluation metrics and diagnostics"
        badge={
          loading
            ? 'Loading Evaluation'
            : status?.model_name ?? 'Model Evaluation'
        }
      />

      {error && (
        <StatusBanner
          variant="warning"
          title="Model data unavailable"
          message={error}
        />
      )}

      {!loading && !error && (
        <StatusBanner
          variant="pending"
          title="Held-out evaluation results"
          message="Metrics shown below come from the existing V2 source-aware evaluation. They describe held-out test performance and should not be interpreted as calibrated confidence or production accuracy."
        />
      )}

      <div className="grid grid-cols-4 gap-4">
        <Card
          title="Active Model"
          subtitle="Currently used by prediction API"
        >
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-[#76b900]/10 border border-[#76b900]/20 flex items-center justify-center">
              <CheckCircle2
                size={18}
                className="text-[#76b900]"
              />
            </div>

            <div>
              <div className="text-[14px] font-semibold text-[#f0f1f3]">
                Random Forest V2
              </div>
              <div className="text-[11px] text-[#7f8490]">
                Loaded
              </div>
            </div>
          </div>
        </Card>

        <Card
          title="Material"
          subtitle="Model domain"
        >
          <div className="text-[14px] font-semibold text-[#f0f1f3]">
            Ti-6Al-4V
          </div>
          <div className="text-[11px] text-[#7f8490] mt-1">
            Laser Powder Bed Fusion
          </div>
        </Card>

        <Card
          title="Dataset"
          subtitle="Experimental records"
        >
          <div className="flex items-center gap-2">
            <Database
              size={16}
              className="text-[#76b900]"
            />
            <span className="text-[20px] font-semibold text-[#f0f1f3] font-mono">
              {status?.dataset_size ?? '—'}
            </span>
            <span className="text-[11px] text-[#7f8490]">
              records
            </span>
          </div>
        </Card>

        <Card
          title="Evaluation Set"
          subtitle="Held-out records"
        >
          <div className="text-[20px] font-semibold text-[#f0f1f3] font-mono">
            {status?.evaluation_samples ?? '—'}
          </div>
          <div className="text-[11px] text-[#7f8490] mt-1">
            source-aware evaluation
          </div>
        </Card>
      </div>

      <Card
        title="Model Evaluation"
        subtitle="R², MAE and RMSE on the existing V2 held-out evaluation"
        headerRight={
          <button
            onClick={loadModelData}
            disabled={loading}
            className="h-8 px-3 rounded-md border border-[#292d35] text-[11px] text-[#9ca1ac] hover:text-[#f0f1f3] hover:border-[#3a3e48] transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            <RefreshCw
              size={13}
              className={
                loading ? 'animate-spin' : ''
              }
            />
            Refresh
          </button>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-[#25282f]">
                <th className="text-left py-3 px-3 text-[10px] uppercase tracking-wider text-[#666b76] font-medium">
                  Model
                </th>

                {TARGETS.map((target) => (
                  <th
                    key={target.id}
                    colSpan={3}
                    className="text-center py-3 px-3 text-[10px] uppercase tracking-wider text-[#666b76] font-medium border-l border-[#25282f]"
                  >
                    {target.label}
                  </th>
                ))}
              </tr>

              <tr className="border-b border-[#25282f]">
                <th />

                {TARGETS.map((target) => (
                  <th
                    key={`${target.id}-metrics`}
                    colSpan={3}
                    className="border-l border-[#25282f] pb-2"
                  >
                    <div className="grid grid-cols-3 text-[9px] text-[#4f545e] uppercase tracking-wider">
                      <span>R²</span>
                      <span>MAE</span>
                      <span>RMSE</span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>

            <tbody>
              {MODEL_ORDER.map((model) => (
                <tr
                  key={model.id}
                  className="border-b border-[#1e2127] last:border-0"
                >
                  <td className="py-4 px-3">
                    <div className="flex items-center gap-3">
                      <span className="h-7 min-w-7 px-1 rounded bg-[#20242b] border border-[#2b2f37] flex items-center justify-center text-[9px] font-semibold text-[#9ca1ac]">
                        {model.abbr}
                      </span>

                      <div>
                        <div className="text-[12px] font-medium text-[#d9dce1]">
                          {model.label}
                        </div>
                        <div className="text-[10px] text-[#626771]">
                          {model.id}
                        </div>
                      </div>
                    </div>
                  </td>

                  {TARGETS.map((target) => {
                    const metric = getMetric(
                      metrics,
                      model.id,
                      target.id
                    );

                    return (
                      <td
                        key={`${model.id}-${target.id}`}
                        className="border-l border-[#25282f] px-3 py-4"
                      >
                        <div className="grid grid-cols-3 gap-2 text-center font-mono text-[11px]">
                          <span
                            className={
                              metric?.r2 !== undefined &&
                              metric?.r2 !== null &&
                              metric.r2 < 0
                                ? 'text-[#d69b62]'
                                : 'text-[#b9bec7]'
                            }
                          >
                            {formatMetric(
                              metric?.r2 ?? null
                            )}
                          </span>

                          <span className="text-[#b9bec7]">
                            {formatMetric(
                              metric?.mae ?? null
                            )}
                          </span>

                          <span className="text-[#b9bec7]">
                            {formatMetric(
                              metric?.rmse ?? null
                            )}
                          </span>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card
        title="Random Forest Feature Importance"
        subtitle="Mean impurity-based importance across the three prediction targets"
      >
        {rfFeatureImportance.length > 0 ? (
          <div className="space-y-4 py-2">
            {rfFeatureImportance.map(
              ({ feature, importance }) => {
                const width =
                  (importance /
                    maxFeatureImportance) *
                  100;

                return (
                  <div
                    key={feature}
                    className="grid grid-cols-[150px_1fr_55px] items-center gap-4"
                  >
                    <div className="text-[11px] text-[#8a8f99] text-right">
                      {FEATURE_LABELS[feature] ??
                        feature}
                    </div>

                    <div className="h-2 bg-[#1f2229] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#76b900] rounded-full"
                        style={{
                          width: `${width}%`,
                        }}
                      />
                    </div>

                    <div className="text-[11px] text-[#aeb3bc] font-mono text-right">
                      {(
                        importance * 100
                      ).toFixed(1)}
                      %
                    </div>
                  </div>
                );
              }
            )}
          </div>
        ) : (
          <div className="py-8 text-center text-[12px] text-[#555a64]">
            Feature importance unavailable.
          </div>
        )}
      </Card>

      <div className="grid grid-cols-2 gap-4">
        <Card
          title="Predicted vs. Experimental"
          subtitle="Random Forest V2 — UTS on held-out evaluation records"
        >
          {parityData.length > 0 ? (
            <div className="h-[300px] relative">
              <svg
                viewBox="0 0 600 300"
                className="w-full h-full"
              >
                <line
                  x1="60"
                  y1="240"
                  x2="560"
                  y2="40"
                  stroke="#343841"
                  strokeWidth="1"
                  strokeDasharray="5 5"
                />

                {parityData.map(
                  (point, index) => {
                    const actualMin = Math.min(
                      ...parityData.map(
                        (p) => p.actual
                      )
                    );

                    const actualMax = Math.max(
                      ...parityData.map(
                        (p) => p.actual
                      )
                    );

                    const min =
                      actualMin - 50;
                    const max =
                      actualMax + 50;

                    const x =
                      60 +
                      ((point.actual - min) /
                        (max - min)) *
                        500;

                    const y =
                      240 -
                      ((point.predicted - min) /
                        (max - min)) *
                        200;

                    return (
                      <circle
                        key={index}
                        cx={x}
                        cy={y}
                        r="3.5"
                        fill="#76b900"
                        opacity="0.75"
                      />
                    );
                  }
                )}

                <text
                  x="310"
                  y="292"
                  textAnchor="middle"
                  fill="#666b76"
                  fontSize="10"
                >
                  Experimental UTS (MPa)
                </text>

                <text
                  x="15"
                  y="145"
                  textAnchor="middle"
                  fill="#666b76"
                  fontSize="10"
                  transform="rotate(-90 15 145)"
                >
                  Predicted UTS (MPa)
                </text>
              </svg>
            </div>
          ) : (
            <div className="py-12 text-center text-[12px] text-[#555a64]">
              Held-out prediction data unavailable.
            </div>
          )}
        </Card>

        <Card
          title="Residual Analysis"
          subtitle="Random Forest V2 — predicted minus experimental UTS"
        >
          {residualData.length > 0 ? (
            <div className="h-[300px] relative">
              <svg
                viewBox="0 0 600 300"
                className="w-full h-full"
              >
                <line
                  x1="60"
                  y1="150"
                  x2="560"
                  y2="150"
                  stroke="#343841"
                  strokeWidth="1"
                  strokeDasharray="5 5"
                />

                {residualData.map(
                  (point, index) => {
                    const actualMin = Math.min(
                      ...residualData.map(
                        (p) => p.actual
                      )
                    );

                    const actualMax = Math.max(
                      ...residualData.map(
                        (p) => p.actual
                      )
                    );

                    const residualMax =
                      Math.max(
                        ...residualData.map(
                          (p) =>
                            Math.abs(
                              p.residual
                            )
                        )
                      ) || 1;

                    const x =
                      60 +
                      ((point.actual -
                        actualMin) /
                        (actualMax -
                          actualMin ||
                          1)) *
                        500;

                    const y =
                      150 -
                      (point.residual /
                        residualMax) *
                        100;

                    return (
                      <circle
                        key={index}
                        cx={x}
                        cy={y}
                        r="3.5"
                        fill="#76b900"
                        opacity="0.75"
                      />
                    );
                  }
                )}

                <text
                  x="310"
                  y="292"
                  textAnchor="middle"
                  fill="#666b76"
                  fontSize="10"
                >
                  Experimental UTS (MPa)
                </text>

                <text
                  x="15"
                  y="145"
                  textAnchor="middle"
                  fill="#666b76"
                  fontSize="10"
                  transform="rotate(-90 15 145)"
                >
                  Residual (MPa)
                </text>
              </svg>
            </div>
          ) : (
            <div className="py-12 text-center text-[12px] text-[#555a64]">
              Residual data unavailable.
            </div>
          )}
        </Card>
      </div>

      <div className="rounded-lg border border-[#3b3225] bg-[#1d1913] px-4 py-3">
        <div className="flex items-start gap-3">
          <AlertTriangle
            size={16}
            className="text-[#d69b62] mt-0.5 shrink-0"
          />

          <div>
            <div className="text-[11px] font-medium text-[#d5b58f]">
              Evaluation interpretation
            </div>

            <p className="text-[11px] text-[#8f8679] mt-1 leading-relaxed">
              The displayed metrics come from the existing
              source-aware held-out V2 evaluation. Negative R²
              values indicate that the corresponding model
              predictions did not outperform the evaluation
              baseline for those targets. These results should
              therefore be treated as model-development
              diagnostics rather than evidence of validated
              production performance.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}