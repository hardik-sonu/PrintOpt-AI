'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  ArrowRight,
  CheckCircle2,
  Cpu,
  Database,
  FlaskConical,
  Layers,
  RefreshCw,
  Zap,
} from 'lucide-react';

import { PageHeader } from '@/components/layout/PageHeader';
import { MetricCard } from '@/components/ui/MetricCard';
import { Card } from '@/components/ui/Card';
import { StatusBanner } from '@/components/layout/StatusBanner';

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ??
  'http://127.0.0.1:8000';

interface DatasetRecord {
  record_id: string;
  source: string;
}

interface SourceRecord {
  id: string;
  record_count: number;
}

interface ModelStatus {
  is_connected: boolean;
  models_available: string[];
  active_model?: string;
  model_name?: string;
  material?: string;
  process?: string;
  dataset_size?: number;
  evaluation_samples?: number;
  evaluation_type?: string;
}

const WORKFLOW_STEPS = [
  {
    label: 'Experimental\nData',
    icon: FlaskConical,
    desc: 'Verified dataset',
  },
  {
    label: 'Process\nParameters',
    icon: Layers,
    desc: 'P · v · t · h',
  },
  {
    label: 'Energy\nInput',
    icon: Zap,
    desc: 'Volumetric energy density',
  },
  {
    label: 'ML\nPrediction',
    icon: Cpu,
    desc: 'Property estimation',
  },
  {
    label: 'Process\nOptimization',
    icon: Cpu,
    desc: 'Constrained search',
  },
  {
    label: 'Recommended\nParameters',
    icon: Layers,
    desc: 'Optimized P · v · t · h',
  },
];

const PHYSICS_CHAIN = [
  'Process Parameters',
  'Energy Input (VED)',
  'Melting / Melt Pool',
  'Solidification',
  'Microstructure',
  'Defects',
  'Material Properties',
];

export default function OverviewPage() {
  const [records, setRecords] = useState<DatasetRecord[]>([]);
  const [sources, setSources] = useState<SourceRecord[]>([]);
  const [modelStatus, setModelStatus] =
    useState<ModelStatus | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadOverview() {
    setLoading(true);
    setError(null);

    try {
      const [datasetResponse, sourcesResponse, modelResponse] =
        await Promise.all([
          fetch(`${API_BASE}/api/dataset`, {
            cache: 'no-store',
          }),
          fetch(`${API_BASE}/api/sources`, {
            cache: 'no-store',
          }),
          fetch(`${API_BASE}/api/model/status`, {
            cache: 'no-store',
          }),
        ]);

      if (!datasetResponse.ok) {
        throw new Error(
          `Dataset API returned HTTP ${datasetResponse.status}`
        );
      }

      if (!sourcesResponse.ok) {
        throw new Error(
          `Sources API returned HTTP ${sourcesResponse.status}`
        );
      }

      if (!modelResponse.ok) {
        throw new Error(
          `Model API returned HTTP ${modelResponse.status}`
        );
      }

      const [
        datasetData,
        sourceData,
        modelData,
      ] = await Promise.all([
        datasetResponse.json() as Promise<DatasetRecord[]>,
        sourcesResponse.json() as Promise<SourceRecord[]>,
        modelResponse.json() as Promise<ModelStatus>,
      ]);

      if (!Array.isArray(datasetData)) {
        throw new Error(
          'Invalid dataset response received.'
        );
      }

      if (!Array.isArray(sourceData)) {
        throw new Error(
          'Invalid source response received.'
        );
      }

      setRecords(datasetData);
      setSources(sourceData);
      setModelStatus(modelData);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to load overview data.'
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadOverview();
  }, []);

  const totalRecords = records.length;

  const totalSources = useMemo(() => {
    return sources.length;
  }, [sources]);

  const activeModel = modelStatus?.model_name
    ?? modelStatus?.active_model
    ?? 'Unavailable';

  const availableModels =
    modelStatus?.models_available?.length ?? 0;

  const modelConnected =
    modelStatus?.is_connected ?? false;

  const material =
    modelStatus?.material ?? 'Ti-6Al-4V';

  const process =
    modelStatus?.process ??
    'Laser Powder Bed Fusion';

  return (
    <div className="space-y-6">
      <PageHeader
        title="Overview"
        subtitle="AI-driven process optimization for LPBF Ti-6Al-4V"
        badge={
          loading
            ? 'Loading'
            : modelConnected
              ? 'System Connected'
              : 'System Offline'
        }
        actions={
          <button
            onClick={loadOverview}
            disabled={loading}
            className="h-8 px-3 rounded-md border border-[#292d35] text-[11px] text-[#9ca1ac] hover:text-[#f0f1f3] hover:border-[#3a3e48] transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            <RefreshCw
              size={13}
              className={
                loading
                  ? 'animate-spin'
                  : ''
              }
            />
            Refresh
          </button>
        }
      />

      {error && (
        <StatusBanner
          variant="warning"
          title="Overview data unavailable"
          message={error}
        />
      )}

      {!loading && !error && (
        <StatusBanner
          variant="pending"
          title="Live project status"
          message={`${material} · ${process} · ${activeModel} connected to the local ML API.`}
        />
      )}

      {/* Hero */}
      <div className="bg-[#12141a] border border-[#1f2229] rounded p-6">
        <div className="flex items-start justify-between gap-6">
          <div>
            <p className="text-[11px] font-medium text-[#76B900] uppercase tracking-widest mb-2">
              PrintOpt AI
            </p>

            <h2 className="text-2xl font-semibold text-[#f0f2f5] leading-tight">
              Optimize the process.
              <br />
              <span className="text-[#76B900]">
                Predict the properties.
              </span>
            </h2>

            <p className="mt-3 text-[13px] text-[#6b7280] max-w-xl leading-relaxed">
              Machine learning–based process parameter
              optimization for Laser Powder Bed Fusion
              of Ti-6Al-4V. From experimental data to
              predicted material properties and
              recommended process parameters.
            </p>
          </div>

          <div className="hidden md:flex flex-col items-end gap-2 text-right">
            <div className="flex items-center gap-2 text-[11px] text-[#7f8490]">
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  modelConnected
                    ? 'bg-[#76B900]'
                    : 'bg-[#8b909a]'
                }`}
              />

              {modelConnected
                ? 'ML API connected'
                : 'ML API unavailable'}
            </div>

            <div className="text-[10px] text-[#454a54] uppercase tracking-wider">
              {activeModel}
            </div>
          </div>
        </div>
      </div>

      {/* Live stats */}
      <div className="grid grid-cols-4 gap-3">
        <MetricCard
          label="Experimental Records"
          value={loading ? '—' : totalRecords}
          sublabel={`${material} LPBF`}
          accent
        />

        <MetricCard
          label="Research Sources"
          value={loading ? '—' : totalSources}
          sublabel="Dataset provenance"
        />

        <MetricCard
          label="Process Parameters"
          value={4}
          sublabel="P · v · t · h (core)"
        />

        <MetricCard
          label="Prediction Targets"
          value={3}
          sublabel="UTS · YS · Elongation"
        />
      </div>

      {/* System status */}
      <Card
        title="System Status"
        subtitle="Current data and model connectivity"
      >
        <div className="grid grid-cols-3 gap-3">
          <div className="flex items-center gap-3 bg-[#0f1012] border border-[#1f2229] rounded p-3">
            <div className="flex items-center justify-center h-8 w-8 rounded bg-[#76B900]/10">
              <Database
                size={15}
                className="text-[#76B900]"
              />
            </div>

            <div>
              <div className="text-[11px] text-[#5a5f6b] uppercase tracking-wider">
                Dataset
              </div>

              <div className="flex items-center gap-1.5 mt-0.5">
                <CheckCircle2
                  size={11}
                  className="text-[#76B900]"
                />

                <span className="text-[12px] text-[#c8cdd6]">
                  {loading
                    ? 'Loading'
                    : `${totalRecords} records loaded`}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3 bg-[#0f1012] border border-[#1f2229] rounded p-3">
            <div className="flex items-center justify-center h-8 w-8 rounded bg-[#76B900]/10">
              <Cpu
                size={15}
                className="text-[#76B900]"
              />
            </div>

            <div>
              <div className="text-[11px] text-[#5a5f6b] uppercase tracking-wider">
                Active Model
              </div>

              <div className="text-[12px] text-[#c8cdd6] mt-0.5">
                {loading
                  ? 'Loading'
                  : activeModel}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3 bg-[#0f1012] border border-[#1f2229] rounded p-3">
            <div className="flex items-center justify-center h-8 w-8 rounded bg-[#76B900]/10">
              <FlaskConical
                size={15}
                className="text-[#76B900]"
              />
            </div>

            <div>
              <div className="text-[11px] text-[#5a5f6b] uppercase tracking-wider">
                Available Models
              </div>

              <div className="text-[12px] text-[#c8cdd6] mt-0.5">
                {loading
                  ? 'Loading'
                  : `${availableModels} model pipelines`}
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Workflow */}
      <Card
        title="Engineering Workflow"
        subtitle="Data to recommendation pipeline"
      >
        <div className="flex items-center gap-1 flex-wrap">
          {WORKFLOW_STEPS.map((step, i) => (
            <div
              key={i}
              className="flex items-center gap-1"
            >
              <div className="flex flex-col items-center px-4 py-3 bg-[#0f1012] rounded border border-[#1f2229] min-w-[90px]">
                <step.icon
                  size={14}
                  className="text-[#76B900] mb-1.5"
                />

                <div className="text-[11px] font-medium text-[#c8cdd6] text-center whitespace-pre-line leading-tight">
                  {step.label}
                </div>

                <div className="text-[10px] text-[#3a3d47] text-center mt-1">
                  {step.desc}
                </div>
              </div>

              {i < WORKFLOW_STEPS.length - 1 && (
                <ArrowRight
                  size={12}
                  className="text-[#3a3d47] flex-shrink-0"
                />
              )}
            </div>
          ))}
        </div>
      </Card>

      {/* Physics chain */}
      <Card
        title="Physical Process Chain"
        subtitle="Mechanisms linking process parameters to material properties"
      >
        <div className="flex items-center gap-0 flex-wrap">
          {PHYSICS_CHAIN.map((step, i) => (
            <div
              key={i}
              className="flex items-center"
            >
              <div
                className="px-3 py-2 text-[11px] font-medium rounded"
                style={{
                  background: `rgba(118,185,0,${0.05 + i * 0.01})`,
                  color:
                    i === 0 ||
                    i === PHYSICS_CHAIN.length - 1
                      ? '#76B900'
                      : '#8b909a',
                  border: `1px solid rgba(118,185,0,${0.1 + i * 0.02})`,
                }}
              >
                {step}
              </div>

              {i < PHYSICS_CHAIN.length - 1 && (
                <ArrowRight
                  size={10}
                  className="text-[#2a2d35] mx-1 flex-shrink-0"
                />
              )}
            </div>
          ))}
        </div>
      </Card>

      {/* Quick access */}
      <div className="grid grid-cols-3 gap-3">
        {[
          {
            title: 'Dataset',
            desc: loading
              ? 'Loading experimental records'
              : `Browse ${totalRecords} experimental records`,
            href: '/dataset',
            tag: loading
              ? 'Loading'
              : `${totalRecords} records`,
          },
          {
            title: 'Prediction',
            desc: 'Enter parameters, get property estimates',
            href: '/prediction',
            tag: 'ML inference',
          },
          {
            title: 'Optimization',
            desc: 'Find constrained process parameters',
            href: '/optimization',
            tag: 'Constrained search',
          },
        ].map((item) => (
          <a
            key={item.title}
            href={item.href}
            className="group bg-[#16181c] hover:bg-[#1a1d24] border border-[#1f2229] hover:border-[#76B900]/20 rounded p-4 transition-colors"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="text-[13px] font-semibold text-[#c8cdd6] group-hover:text-[#f0f2f5] transition-colors">
                {item.title}
              </div>

              <ArrowRight
                size={12}
                className="text-[#3a3d47] group-hover:text-[#76B900] transition-colors"
              />
            </div>

            <p className="text-[11px] text-[#5a5f6b]">
              {item.desc}
            </p>

            <div className="mt-3 inline-block px-2 py-0.5 text-[10px] font-medium rounded border border-[#2a2d35] text-[#5a5f6b] uppercase tracking-wider">
              {item.tag}
            </div>
          </a>
        ))}
      </div>

      {/* Technical note */}
      <div className="flex items-center gap-2 text-[11px] text-[#3a3d47]">
        <span>
          Material: {material}
        </span>

        <span>·</span>

        <span>
          Process: {process}
        </span>

        <span>·</span>

        <span>
          Core parameters: P · v · h · t
        </span>

        <span>·</span>

        <span>
          Targets: UTS · YS · Elongation
        </span>
      </div>
    </div>
  );
}