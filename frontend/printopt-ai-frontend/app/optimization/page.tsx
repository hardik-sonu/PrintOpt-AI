'use client';

import { useState } from 'react';
import { PageHeader } from '@/components/layout/PageHeader';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { StatusBanner } from '@/components/layout/StatusBanner';
import { MetricCard } from '@/components/ui/MetricCard';
import { PARAMETER_RANGES } from '@/lib/constants';
import { optimizationApi } from '@/services/api';
import type {
  OptimizationResult,
  OptimizationObjective,
} from '@/types';
import { Target, Zap } from 'lucide-react';
import { calculateVED } from '@/lib/ved';

const OBJECTIVES: {
  id: OptimizationObjective['target'];
  label: string;
  desc: string;
}[] = [
  {
    id: 'maximize_uts',
    label: 'Maximize UTS',
    desc: 'Highest tensile strength',
  },
  {
    id: 'maximize_yield_strength',
    label: 'Maximize Yield Strength',
    desc: 'Highest yield strength',
  },
  {
    id: 'maximize_elongation',
    label: 'Maximize Elongation',
    desc: 'Best ductility',
  },
  {
    id: 'multi_objective',
    label: 'Multi-Objective',
    desc: 'Balanced trade-off',
  },
];

type ConstraintKey =
  | 'laser_power_w'
  | 'scan_speed_mm_s'
  | 'layer_thickness_um'
  | 'hatch_spacing_um';

export default function OptimizationPage() {
  const [objective, setObjective] =
    useState<OptimizationObjective['target']>('maximize_uts');

  const [constraints, setConstraints] = useState({
    laser_power_w: {
      min: PARAMETER_RANGES.laser_power_w.min,
      max: PARAMETER_RANGES.laser_power_w.max,
    },
    scan_speed_mm_s: {
      min: PARAMETER_RANGES.scan_speed_mm_s.min,
      max: PARAMETER_RANGES.scan_speed_mm_s.max,
    },
    layer_thickness_um: {
      min: PARAMETER_RANGES.layer_thickness_um.min,
      max: PARAMETER_RANGES.layer_thickness_um.max,
    },
    hatch_spacing_um: {
      min: PARAMETER_RANGES.hatch_spacing_um.min,
      max: PARAMETER_RANGES.hatch_spacing_um.max,
    },
  });

  const [result, setResult] =
    useState<OptimizationResult | null>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasInvalidConstraints = (
    Object.keys(constraints) as ConstraintKey[]
  ).some(
    (key) =>
      constraints[key].min >= constraints[key].max
  );

  const handleConstraintChange = (
    key: ConstraintKey,
    field: 'min' | 'max',
    value: string
  ) => {
    const range = PARAMETER_RANGES[key];
    const parsedValue = Number(value);

    const safeValue = Number.isFinite(parsedValue)
      ? parsedValue
      : field === 'min'
        ? range.min
        : range.max;

    setConstraints((prev) => ({
      ...prev,
      [key]: {
        ...prev[key],
        [field]: safeValue,
      },
    }));

    setError(null);
  };

  const handleOptimize = async () => {
    setError(null);

    if (hasInvalidConstraints) {
      setResult(null);
      setError(
        'Invalid parameter constraints. Each minimum value must be lower than its maximum value.'
      );
      return;
    }

    setIsLoading(true);
    setResult(null);

    const response = await optimizationApi.optimize({
      objective: {
        target: objective,
      },
      constraints,
    });

    setIsLoading(false);

    if (response.status === 'error') {
      setError(
        response.error ??
          'Optimization failed. Check that the ML service is running and try again.'
      );
      return;
    }

    if (response.data) {
      setResult(response.data);
      return;
    }

    setError(
      'The optimization service returned no optimization result.'
    );
  };

  const resultVED = result
    ? calculateVED(
        result.recommended_parameters.laser_power_w,
        result.recommended_parameters.scan_speed_mm_s,
        result.recommended_parameters.hatch_spacing_um,
        result.recommended_parameters.layer_thickness_um
      )
    : null;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Optimization"
        subtitle="Find optimal process parameters for your target properties"
        badge="Constrained Search"
      />

      {error && (
        <StatusBanner
          variant="warning"
          title="Optimization unavailable"
          message={error}
        />
      )}

      <div className="grid grid-cols-[1fr_1fr] gap-4">
        {/* Left: Setup */}
        <div className="space-y-4">
          <Card title="Optimization Objective">
            <div className="space-y-1.5">
              {OBJECTIVES.map((obj) => (
                <button
                  key={obj.id}
                  type="button"
                  onClick={() => {
                    setObjective(obj.id);
                    setError(null);
                  }}
                  className={`w-full rounded border px-3 py-2.5 text-left text-[12px] transition-colors ${
                    objective === obj.id
                      ? 'border-[#76B900]/40 bg-[#76B900]/8 text-[#76B900]'
                      : 'border-[#1f2229] text-[#8b909a] hover:border-[#2a2d35] hover:text-[#c8cdd6]'
                  }`}
                >
                  <div className="font-medium">
                    {obj.label}
                  </div>

                  <div className="mt-0.5 text-[10px] opacity-70">
                    {obj.desc}
                  </div>
                </button>
              ))}
            </div>
          </Card>

          <Card
            title="Parameter Constraints"
            subtitle="Define the search space bounds"
          >
            <div className="space-y-4">
              {[
                {
                  key: 'laser_power_w' as const,
                  label: 'Laser Power',
                  unit: 'W',
                  range: PARAMETER_RANGES.laser_power_w,
                },
                {
                  key: 'scan_speed_mm_s' as const,
                  label: 'Scan Speed',
                  unit: 'mm/s',
                  range: PARAMETER_RANGES.scan_speed_mm_s,
                },
                {
                  key: 'layer_thickness_um' as const,
                  label: 'Layer Thickness',
                  unit: 'µm',
                  range: PARAMETER_RANGES.layer_thickness_um,
                },
                {
                  key: 'hatch_spacing_um' as const,
                  label: 'Hatch Spacing',
                  unit: 'µm',
                  range: PARAMETER_RANGES.hatch_spacing_um,
                },
              ].map(({ key, label, unit, range }) => {
                const invalid =
                  constraints[key].min >=
                  constraints[key].max;

                return (
                  <div key={key} className="space-y-1">
                    <div className="flex items-center justify-between">
                      <div className="text-[11px] font-medium text-[#8b909a]">
                        {label}
                      </div>

                      {invalid && (
                        <span className="text-[10px] uppercase tracking-wider text-[#d97706]">
                          Invalid range
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-2">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] text-[#5a5f6b]">
                          Min
                        </span>

                        <input
                          type="number"
                          value={constraints[key].min}
                          min={range.min}
                          max={range.max}
                          step={range.step}
                          onChange={(event) =>
                            handleConstraintChange(
                              key,
                              'min',
                              event.target.value
                            )
                          }
                          className={`w-20 rounded border bg-[#0f1012] px-2 py-1 text-right text-[12px] text-[#f0f2f5] focus:outline-none ${
                            invalid
                              ? 'border-[#d97706]/60'
                              : 'border-[#2a2d35] focus:border-[#76B900]/50'
                          }`}
                        />
                      </div>

                      <span className="text-[10px] text-[#3a3d47]">
                        –
                      </span>

                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] text-[#5a5f6b]">
                          Max
                        </span>

                        <input
                          type="number"
                          value={constraints[key].max}
                          min={range.min}
                          max={range.max}
                          step={range.step}
                          onChange={(event) =>
                            handleConstraintChange(
                              key,
                              'max',
                              event.target.value
                            )
                          }
                          className={`w-20 rounded border bg-[#0f1012] px-2 py-1 text-right text-[12px] text-[#f0f2f5] focus:outline-none ${
                            invalid
                              ? 'border-[#d97706]/60'
                              : 'border-[#2a2d35] focus:border-[#76B900]/50'
                          }`}
                        />

                        <span className="text-[11px] text-[#5a5f6b]">
                          {unit}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            <Button
              variant="primary"
              size="lg"
              className="mt-5 w-full"
              isLoading={isLoading}
              disabled={hasInvalidConstraints}
              onClick={handleOptimize}
            >
              <Target size={14} />
              Run Optimization
            </Button>
          </Card>
        </div>

        {/* Right: Results */}
        <div className="space-y-4">
          <Card
            title="Recommended Parameters"
            subtitle="Optimal parameters from the ML optimizer"
          >
            {result ? (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <MetricCard
                    label="Laser Power"
                    value={
                      result.recommended_parameters
                        .laser_power_w
                    }
                    unit="W"
                    accent
                  />

                  <MetricCard
                    label="Scan Speed"
                    value={
                      result.recommended_parameters
                        .scan_speed_mm_s
                    }
                    unit="mm/s"
                  />

                  <MetricCard
                    label="Layer Thickness"
                    value={
                      result.recommended_parameters
                        .layer_thickness_um
                    }
                    unit="µm"
                  />

                  <MetricCard
                    label="Hatch Spacing"
                    value={
                      result.recommended_parameters
                        .hatch_spacing_um
                    }
                    unit="µm"
                  />
                </div>

                {resultVED !== null && (
                  <div className="flex items-center gap-2 rounded border border-[#1f2229] bg-[#0f1012] px-3 py-2">
                    <Zap
                      size={12}
                      className="text-[#76B900]"
                    />

                    <span className="text-[11px] text-[#5a5f6b]">
                      VED
                    </span>

                    <span className="ml-auto text-[13px] font-semibold text-[#76B900]">
                      {resultVED.toFixed(1)}
                    </span>

                    <span className="text-[11px] text-[#5a5f6b]">
                      J/mm³
                    </span>
                  </div>
                )}

                <div className="border-t border-[#1a1d24] pt-3 text-[11px] text-[#3a3d47]">
                  Objective:{' '}
                  {
                    OBJECTIVES.find(
                      (item) => item.id === objective
                    )?.label
                  }
                </div>
              </div>
            ) : (
              <div className="py-8 text-center">
                <Target
                  size={24}
                  className="mx-auto mb-3 text-[#2a2d35]"
                />

                <p className="text-[12px] text-[#3a3d47]">
                  Awaiting optimization
                </p>

                <p className="mt-1 text-[11px] text-[#2a2d35]">
                  Select an objective, define the parameter
                  bounds, and run the optimizer
                </p>
              </div>
            )}
          </Card>

          <Card
            title="Predicted Properties"
            subtitle="At recommended parameters"
          >
            {result ? (
              <div className="space-y-3">
                <MetricCard
                  label="Ultimate Tensile Strength"
                  value={
                    result.predicted_properties
                      .uts_mpa?.toFixed(0) ?? '—'
                  }
                  unit="MPa"
                  accent
                />

                <MetricCard
                  label="Yield Strength"
                  value={
                    result.predicted_properties
                      .yield_strength_mpa?.toFixed(0) ??
                    '—'
                  }
                  unit="MPa"
                />

                <MetricCard
                  label="Elongation"
                  value={
                    result.predicted_properties
                      .elongation_pct?.toFixed(1) ?? '—'
                  }
                  unit="%"
                />

                {result.confidence != null && (
                  <div className="border-t border-[#1a1d24] pt-2 text-[11px] text-[#5a5f6b]">
                    Confidence:{' '}
                    {(result.confidence * 100).toFixed(0)}%
                  </div>
                )}
              </div>
            ) : (
              <div className="py-6 text-center">
                <p className="text-[12px] text-[#3a3d47]">
                  No results yet
                </p>

                <p className="mt-1 text-[11px] text-[#2a2d35]">
                  Predicted properties will appear after
                  optimization
                </p>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}