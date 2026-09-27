'use client';

import { useState, useCallback } from 'react';
import { PageHeader } from '@/components/layout/PageHeader';
import { Card } from '@/components/ui/Card';
import { ParameterInput } from '@/components/ui/ParameterInput';
import { VEDDisplay } from '@/components/ui/VEDDisplay';
import { Button } from '@/components/ui/Button';
import { StatusBanner } from '@/components/layout/StatusBanner';
import { MetricCard } from '@/components/ui/MetricCard';
import { calculateVED } from '@/lib/ved';
import { PARAMETER_RANGES } from '@/lib/constants';
import { predictionApi } from '@/services/api';
import type { PredictionResult } from '@/types';
import { BrainCircuit, Database, CheckCircle2 } from 'lucide-react';

const UNSUPPORTED_PROPERTIES = [
  'Relative Density',
  'Porosity',
  'Vickers Hardness',
  'Surface Roughness',
];

export default function PredictionPage() {
  const [params, setParams] = useState({
    laser_power_w: PARAMETER_RANGES.laser_power_w.default,
    scan_speed_mm_s: PARAMETER_RANGES.scan_speed_mm_s.default,
    layer_thickness_um: PARAMETER_RANGES.layer_thickness_um.default,
    hatch_spacing_um: PARAMETER_RANGES.hatch_spacing_um.default,
    laser_spot_um: PARAMETER_RANGES.laser_spot_um.default,
    powder_size_um: PARAMETER_RANGES.powder_size_um.default,
  });

  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ved = calculateVED(
    params.laser_power_w,
    params.scan_speed_mm_s,
    params.hatch_spacing_um,
    params.layer_thickness_um
  );

  const setParam = useCallback(
    (key: keyof typeof params) => (value: number) => {
      setParams((prev) => ({
        ...prev,
        [key]: value,
      }));
    },
    []
  );

  const handlePredict = async () => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    const response = await predictionApi.predict(params);

    setIsLoading(false);

    if (response.status === 'error') {
      setError(
        response.error ??
          'Prediction failed. Check that the ML service is running and try again.'
      );
      return;
    }

    if (response.data) {
      setResult(response.data);
    } else {
      setError('The ML service returned no prediction data.');
    }
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="Prediction"
        subtitle="Input process parameters to get ML-predicted mechanical properties"
        badge="ML Inference"
      />

      {error && (
        <StatusBanner
          variant="warning"
          title="Prediction unavailable"
          message={error}
        />
      )}

      <div className="grid grid-cols-[1fr_1fr] gap-4">
        {/* Inputs */}
        <Card
          title="Process Parameters"
          subtitle="Core inputs to the prediction model"
        >
          <div className="space-y-5">
            <ParameterInput
              label="Laser Power"
              unit="W"
              value={params.laser_power_w}
              min={PARAMETER_RANGES.laser_power_w.min}
              max={PARAMETER_RANGES.laser_power_w.max}
              step={PARAMETER_RANGES.laser_power_w.step}
              onChange={setParam('laser_power_w')}
            />

            <ParameterInput
              label="Scan Speed"
              unit="mm/s"
              value={params.scan_speed_mm_s}
              min={PARAMETER_RANGES.scan_speed_mm_s.min}
              max={PARAMETER_RANGES.scan_speed_mm_s.max}
              step={PARAMETER_RANGES.scan_speed_mm_s.step}
              onChange={setParam('scan_speed_mm_s')}
            />

            <ParameterInput
              label="Layer Thickness"
              unit="µm"
              value={params.layer_thickness_um}
              min={PARAMETER_RANGES.layer_thickness_um.min}
              max={PARAMETER_RANGES.layer_thickness_um.max}
              step={PARAMETER_RANGES.layer_thickness_um.step}
              onChange={setParam('layer_thickness_um')}
            />

            <ParameterInput
              label="Hatch Spacing"
              unit="µm"
              value={params.hatch_spacing_um}
              min={PARAMETER_RANGES.hatch_spacing_um.min}
              max={PARAMETER_RANGES.hatch_spacing_um.max}
              step={PARAMETER_RANGES.hatch_spacing_um.step}
              onChange={setParam('hatch_spacing_um')}
            />

            <div className="border-t border-[#1f2229] pt-4">
              <div className="mb-3 text-[11px] uppercase tracking-wider text-[#5a5f6b]">
                Optional Parameters
              </div>

              <div className="space-y-4">
                <ParameterInput
                  label="Laser Spot Diameter"
                  unit="µm"
                  value={params.laser_spot_um}
                  min={PARAMETER_RANGES.laser_spot_um.min}
                  max={PARAMETER_RANGES.laser_spot_um.max}
                  step={PARAMETER_RANGES.laser_spot_um.step}
                  onChange={setParam('laser_spot_um')}
                />

                <ParameterInput
                  label="Powder Size D50"
                  unit="µm"
                  value={params.powder_size_um}
                  min={PARAMETER_RANGES.powder_size_um.min}
                  max={PARAMETER_RANGES.powder_size_um.max}
                  step={PARAMETER_RANGES.powder_size_um.step}
                  onChange={setParam('powder_size_um')}
                />
              </div>
            </div>

            <VEDDisplay ved={ved} />

            <Button
              variant="primary"
              size="lg"
              className="w-full"
              isLoading={isLoading}
              onClick={handlePredict}
            >
              <BrainCircuit size={15} />
              Run Prediction
            </Button>
          </div>
        </Card>

        {/* Results */}
        <div className="space-y-4">
          <Card
            title="Predicted Properties"
            subtitle="Results from the connected ML model"
          >
            {result ? (
              <div className="space-y-3">
                {result.coverage_warning && (
                  <StatusBanner
                    variant="warning"
                    title="Coverage warning"
                    message={result.coverage_warning}
                  />
                )}

                <div className="grid grid-cols-1 gap-3">
                  <MetricCard
                    label="Ultimate Tensile Strength"
                    value={result.uts_mpa?.toFixed(0) ?? '—'}
                    unit="MPa"
                    sublabel={
                      result.uts_uncertainty != null
                        ? `± ${result.uts_uncertainty.toFixed(0)} MPa`
                        : undefined
                    }
                    accent
                  />

                  <MetricCard
                    label="Yield Strength"
                    value={result.yield_strength_mpa?.toFixed(0) ?? '—'}
                    unit="MPa"
                    sublabel={
                      result.yield_strength_uncertainty != null
                        ? `± ${result.yield_strength_uncertainty.toFixed(0)} MPa`
                        : undefined
                    }
                  />

                  <MetricCard
                    label="Elongation"
                    value={result.elongation_pct?.toFixed(1) ?? '—'}
                    unit="%"
                    sublabel={
                      result.elongation_uncertainty != null
                        ? `± ${result.elongation_uncertainty.toFixed(1)}%`
                        : undefined
                    }
                  />
                </div>

                <div className="mt-2 border-t border-[#1a1d24] pt-3 text-[11px] text-[#3a3d47]">
                  VED: {result.ved_j_mm3.toFixed(1)} J/mm³
                  {result.model_version && ` · Model: ${result.model_version}`}
                </div>
              </div>
            ) : (
              <div className="py-8 text-center">
                <BrainCircuit
                  size={24}
                  className="mx-auto mb-3 text-[#2a2d35]"
                />

                <p className="text-[12px] text-[#3a3d47]">
                  Awaiting prediction
                </p>

                <p className="mt-1 text-[11px] text-[#2a2d35]">
                  Set process parameters and run the ML model
                </p>
              </div>
            )}
          </Card>

          <Card
            title="Model Coverage"
            subtitle="Properties currently supported by the available dataset"
          >
            <div className="space-y-3">
              {/* Supported outputs */}
              <div className="rounded-md border border-[#1c241d] bg-[#101410] p-3">
                <div className="mb-2 flex items-center gap-2">
                  <CheckCircle2
                    size={14}
                    className="text-[#76B900]"
                  />

                  <span className="text-[11px] font-medium uppercase tracking-wider text-[#76B900]">
                    Supported Outputs
                  </span>
                </div>

                <div className="space-y-1.5">
                  {[
                    'Ultimate Tensile Strength',
                    'Yield Strength',
                    'Elongation',
                  ].map((item) => (
                    <div
                      key={item}
                      className="flex items-center gap-2 py-1"
                    >
                      <span className="h-1.5 w-1.5 rounded-full bg-[#76B900]" />

                      <span className="text-[12px] text-[#7c827d]">
                        {item}
                      </span>

                      <span className="ml-auto text-[10px] uppercase tracking-wider text-[#4d554e]">
                        ML
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Unsupported outputs */}
              <div className="rounded-md border border-[#1a1d24] bg-[#0d0f12] p-3">
                <div className="mb-2 flex items-center gap-2">
                  <Database
                    size={14}
                    className="text-[#555b65]"
                  />

                  <span className="text-[11px] font-medium uppercase tracking-wider text-[#555b65]">
                    Dataset Expansion Required
                  </span>
                </div>

                <p className="mb-3 text-[11px] leading-relaxed text-[#3f444d]">
                  These properties are not predicted because the current
                  modeling dataset does not provide supported target data for
                  these outputs.
                </p>

                <div className="space-y-1.5">
                  {UNSUPPORTED_PROPERTIES.map((item) => (
                    <div
                      key={item}
                      className="flex items-center gap-2 border-b border-[#171a20] py-1.5 last:border-0"
                    >
                      <span className="h-1.5 w-1.5 rounded-full bg-[#2a2d35]" />

                      <span className="text-[12px] text-[#3a3d47]">
                        {item}
                      </span>

                      <span className="ml-auto text-[10px] uppercase tracking-wider text-[#2a2d35]">
                        Not available
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}