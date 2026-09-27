'use client';

import { useEffect, useMemo, useState } from 'react';
import { PageHeader } from '@/components/layout/PageHeader';
import { Card } from '@/components/ui/Card';
import { StatusBanner } from '@/components/layout/StatusBanner';
import { processMapApi } from '@/services/api';
import type { ProcessMapData } from '@/types';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Label,
} from 'recharts';
import { Database, Loader2, RefreshCw } from 'lucide-react';

interface ScatterPoint {
  x: number;
  y: number;
  id: string;
  source: string;
}

interface TooltipProps {
  active?: boolean;
  payload?: {
    payload: ScatterPoint;
  }[];
}

function ScatterTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null;

  const d = payload[0].payload;

  return (
    <div className="bg-[#1d2026] border border-[#2a2d35] rounded px-3 py-2.5 text-[11px] shadow-xl">
      <div className="font-mono text-[#76B900] mb-1">
        {d.id}
      </div>

      <div className="text-[#8b909a]">
        {d.source}
      </div>

      <div className="mt-1 space-y-0.5">
        <div>
          <span className="text-[#5a5f6b]">VED: </span>
          <span className="text-[#c8cdd6]">
            {d.x.toFixed(1)} J/mm³
          </span>
        </div>

        <div>
          <span className="text-[#5a5f6b]">Value: </span>
          <span className="text-[#c8cdd6]">
            {d.y}
          </span>
        </div>
      </div>
    </div>
  );
}

const CHART_STYLE = {
  background: '#0f1012',
  fontSize: 11,
};

interface ScatterPlotProps {
  title: string;
  subtitle: string;
  data: ScatterPoint[];
  yLabel: string;
  yUnit: string;
}

function ScatterPlot({
  title,
  subtitle,
  data,
  yLabel,
  yUnit,
}: ScatterPlotProps) {
  return (
    <Card title={title} subtitle={subtitle}>
      {data.length > 0 ? (
        <ResponsiveContainer width="100%" height={260}>
          <ScatterChart
            style={CHART_STYLE}
            margin={{
              top: 10,
              right: 16,
              bottom: 32,
              left: 16,
            }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#1f2229"
            />

            <XAxis
              dataKey="x"
              type="number"
              tick={{
                fill: '#5a5f6b',
                fontSize: 10,
              }}
              axisLine={{
                stroke: '#2a2d35',
              }}
              tickLine={false}
            >
              <Label
                value="VED (J/mm³)"
                offset={-10}
                position="insideBottom"
                fill="#5a5f6b"
                fontSize={10}
              />
            </XAxis>

            <YAxis
              dataKey="y"
              type="number"
              tick={{
                fill: '#5a5f6b',
                fontSize: 10,
              }}
              axisLine={{
                stroke: '#2a2d35',
              }}
              tickLine={false}
            >
              <Label
                value={`${yLabel} (${yUnit})`}
                angle={-90}
                position="insideLeft"
                fill="#5a5f6b"
                fontSize={10}
              />
            </YAxis>

            <Tooltip
              content={<ScatterTooltip />}
            />

            <Scatter
              data={data}
              fill="#76B900"
              fillOpacity={0.7}
              r={4}
            />
          </ScatterChart>
        </ResponsiveContainer>
      ) : (
        <div className="h-[260px] flex items-center justify-center">
          <div className="text-center">
            <Database
              size={22}
              className="text-[#2a2d35] mx-auto mb-2"
            />
            <p className="text-[12px] text-[#5a5f6b]">
              No data available
            </p>
          </div>
        </div>
      )}
    </Card>
  );
}

export default function ProcessMapPage() {
  const [data, setData] = useState<ProcessMapData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadProcessMap = async () => {
    setIsLoading(true);
    setError(null);

    const response = await processMapApi.getData();

    if (response.status === 'error' || !response.data) {
      setError(
        response.error ??
          'Unable to load process map data.'
      );
      setData(null);
    } else {
      setData(response.data);
    }

    setIsLoading(false);
  };

  useEffect(() => {
    loadProcessMap();
  }, []);

  const utsData = useMemo<ScatterPoint[]>(() => {
    if (!data) return [];

    return data.ved_values
      .map((ved, index) => ({
        x: ved,
        y: data.uts_values[index],
        id: data.record_ids[index] ?? `Record ${index + 1}`,
        source: data.sources[index] ?? 'Unknown source',
      }))
      .filter(
        (point): point is ScatterPoint =>
          point.y !== null &&
          point.y !== undefined &&
          Number.isFinite(point.x) &&
          Number.isFinite(point.y)
      );
  }, [data]);

  const ysData = useMemo<ScatterPoint[]>(() => {
    if (!data) return [];

    return data.ved_values
      .map((ved, index) => ({
        x: ved,
        y: data.yield_strength_values[index],
        id: data.record_ids[index] ?? `Record ${index + 1}`,
        source: data.sources[index] ?? 'Unknown source',
      }))
      .filter(
        (point): point is ScatterPoint =>
          point.y !== null &&
          point.y !== undefined &&
          Number.isFinite(point.x) &&
          Number.isFinite(point.y)
      );
  }, [data]);

  const elData = useMemo<ScatterPoint[]>(() => {
    if (!data) return [];

    return data.ved_values
      .map((ved, index) => ({
        x: ved,
        y: data.elongation_values[index],
        id: data.record_ids[index] ?? `Record ${index + 1}`,
        source: data.sources[index] ?? 'Unknown source',
      }))
      .filter(
        (point): point is ScatterPoint =>
          point.y !== null &&
          point.y !== undefined &&
          Number.isFinite(point.x) &&
          Number.isFinite(point.y)
      );
  }, [data]);

  const totalRecords = data?.record_ids.length ?? 0;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Process Map"
        subtitle="VED vs. mechanical property relationships from experimental data"
        badge="Scientific Plots"
      />

      {error && (
        <StatusBanner
          variant="warning"
          title="Process map data unavailable"
          message={error}
        />
      )}

      {isLoading ? (
        <Card>
          <div className="py-16 flex flex-col items-center justify-center">
            <Loader2
              size={22}
              className="text-[#76B900] animate-spin mb-3"
            />

            <p className="text-[12px] text-[#8b909a]">
              Loading experimental process data...
            </p>
          </div>
        </Card>
      ) : (
        <>
          {data && (
            <div className="flex items-center justify-between px-1">
              <div className="flex items-center gap-2">
                <Database
                  size={13}
                  className="text-[#76B900]"
                />

                <span className="text-[11px] text-[#5a5f6b]">
                  Experimental records
                </span>

                <span className="text-[12px] font-semibold text-[#c8cdd6]">
                  {totalRecords}
                </span>
              </div>

              <button
                type="button"
                onClick={loadProcessMap}
                className="flex items-center gap-1.5 text-[11px] text-[#5a5f6b] hover:text-[#76B900] transition-colors"
              >
                <RefreshCw size={12} />
                Refresh
              </button>
            </div>
          )}

          <div className="grid grid-cols-2 gap-4">
            <ScatterPlot
              title="VED vs. Ultimate Tensile Strength"
              subtitle="Effect of energy input on UTS"
              data={utsData}
              yLabel="UTS"
              yUnit="MPa"
            />

            <ScatterPlot
              title="VED vs. Yield Strength"
              subtitle="Effect of energy input on yield strength"
              data={ysData}
              yLabel="YS"
              yUnit="MPa"
            />
          </div>

          <ScatterPlot
            title="VED vs. Elongation"
            subtitle="Effect of energy input on ductility"
            data={elData}
            yLabel="Elongation"
            yUnit="%"
          />

          {data && (
            <div className="text-[11px] text-[#3a3d47] px-1">
              VED = P / (v × h × t) — computed dynamically from
              experimental process parameters. Hover data points for
              record details.
            </div>
          )}
        </>
      )}
    </div>
  );
}