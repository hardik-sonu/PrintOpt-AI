'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  Database,
  RefreshCw,
  Search,
} from 'lucide-react';

import { PageHeader } from '@/components/layout/PageHeader';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { StatusBanner } from '@/components/layout/StatusBanner';

interface DatasetRecord {
  record_id: string;
  laser_power_w: number;
  scan_speed_mm_s: number;
  layer_thickness_um: number;
  hatch_spacing_um: number;
  laser_spot_um?: number;
  powder_size_um?: number;
  ved_j_mm3: number;
  uts_mpa?: number;
  yield_strength_mpa?: number;
  elongation_pct?: number;
  source: string;
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ??
  'https://printopt-ai-production.up.railway.app';

const COLUMNS = [
  {
    key: 'record_id',
    label: 'Record ID',
    align: 'left' as const,
  },
  {
    key: 'laser_power_w',
    label: 'P (W)',
    align: 'right' as const,
  },
  {
    key: 'scan_speed_mm_s',
    label: 'v (mm/s)',
    align: 'right' as const,
  },
  {
    key: 'layer_thickness_um',
    label: 't (µm)',
    align: 'right' as const,
  },
  {
    key: 'hatch_spacing_um',
    label: 'h (µm)',
    align: 'right' as const,
  },
  {
    key: 'laser_spot_um',
    label: 'Spot (µm)',
    align: 'right' as const,
  },
  {
    key: 'powder_size_um',
    label: 'Powder (µm)',
    align: 'right' as const,
  },
  {
    key: 'ved_j_mm3',
    label: 'VED (J/mm³)',
    align: 'right' as const,
  },
  {
    key: 'uts_mpa',
    label: 'UTS (MPa)',
    align: 'right' as const,
  },
  {
    key: 'yield_strength_mpa',
    label: 'YS (MPa)',
    align: 'right' as const,
  },
  {
    key: 'elongation_pct',
    label: 'El. (%)',
    align: 'right' as const,
  },
  {
    key: 'source',
    label: 'Source',
    align: 'left' as const,
  },
];

export default function DatasetPage() {
  const [records, setRecords] = useState<DatasetRecord[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadDataset() {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE}/api/dataset`,
        {
          cache: 'no-store',
        }
      );

      if (!response.ok) {
        throw new Error(
          `HTTP ${response.status}: ${response.statusText}`
        );
      }

      const data =
        (await response.json()) as DatasetRecord[];

      if (!Array.isArray(data)) {
        throw new Error(
          'Invalid dataset response received from API.'
        );
      }

      setRecords(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to load dataset.'
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadDataset();
  }, []);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) {
      return records;
    }

    return records.filter((record) => {
      return (
        record.record_id
          .toString()
          .toLowerCase()
          .includes(query) ||
        record.source
          .toLowerCase()
          .includes(query)
      );
    });
  }, [records, search]);

  const sourceCount = useMemo(() => {
    return new Set(
      records.map((record) => record.source)
    ).size;
  }, [records]);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Dataset"
        subtitle="Experimental LPBF Ti-6Al-4V records from peer-reviewed literature"
        badge={
          loading
            ? 'Loading Dataset'
            : `${records.length} Records`
        }
        actions={
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search
                size={12}
                className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[#5a5f6b]"
              />

              <input
                className="pl-7 pr-3 py-1.5 text-[12px] bg-[#16181c] border border-[#2a2d35] rounded text-[#c8cdd6] placeholder-[#3a3d47] focus:outline-none focus:border-[#76B900]/50 w-52"
                placeholder="Search records or sources..."
                value={search}
                onChange={(e) =>
                  setSearch(e.target.value)
                }
              />
            </div>

            <span className="text-[11px] text-[#5a5f6b]">
              {loading
                ? 'Loading...'
                : `${filtered.length} / ${records.length} records`}
            </span>

            <button
              onClick={loadDataset}
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
          </div>
        }
      />

      {error && (
        <StatusBanner
          variant="warning"
          title="Dataset unavailable"
          message={error}
        />
      )}

      <div className="grid grid-cols-4 gap-3 mb-2">
        <div className="bg-[#16181c] border border-[#1f2229] rounded px-4 py-3">
          <div className="text-[11px] text-[#5a5f6b] uppercase tracking-wider mb-1">
            Total Records
          </div>

          <div className="text-xl font-semibold text-[#f0f2f5]">
            {loading ? '—' : records.length}
          </div>
        </div>

        <div className="bg-[#16181c] border border-[#1f2229] rounded px-4 py-3">
          <div className="text-[11px] text-[#5a5f6b] uppercase tracking-wider mb-1">
            Research Sources
          </div>

          <div className="text-xl font-semibold text-[#f0f2f5]">
            {loading ? '—' : sourceCount}
          </div>
        </div>

        <div className="bg-[#16181c] border border-[#1f2229] rounded px-4 py-3">
          <div className="text-[11px] text-[#5a5f6b] uppercase tracking-wider mb-1">
            Core Parameters
          </div>

          <div className="text-xl font-semibold text-[#f0f2f5]">
            4
          </div>

          <div className="text-[10px] text-[#4b505a] mt-0.5">
            P · v · h · t
          </div>
        </div>

        <div className="bg-[#16181c] border border-[#1f2229] rounded px-4 py-3">
          <div className="text-[11px] text-[#5a5f6b] uppercase tracking-wider mb-1">
            Output Targets
          </div>

          <div className="text-xl font-semibold text-[#f0f2f5]">
            3
          </div>

          <div className="text-[10px] text-[#4b505a] mt-0.5">
            UTS · YS · Elongation
          </div>
        </div>
      </div>

      {!loading && !error && (
        <StatusBanner
          variant="pending"
          title="Experimental dataset"
          message={`Loaded ${records.length} records directly from the verified Ti-6Al-4V LPBF dataset. Values shown below are dataset records, not generated demo values.`}
        />
      )}

      <Card
        title="Experimental Records"
        subtitle="Process parameters and measured mechanical properties"
        headerRight={
          <Badge variant="gray">
            {loading
              ? 'Loading'
              : `${filtered.length} of ${records.length}`}
          </Badge>
        }
      >
        {loading ? (
          <div className="py-16 text-center">
            <RefreshCw
              size={18}
              className="animate-spin text-[#76B900] mx-auto mb-3"
            />

            <div className="text-[12px] text-[#686e79]">
              Loading experimental records...
            </div>
          </div>
        ) : error ? (
          <div className="py-16 text-center">
            <Database
              size={18}
              className="text-[#4f545e] mx-auto mb-3"
            />

            <div className="text-[12px] text-[#686e79]">
              Dataset records could not be loaded.
            </div>
          </div>
        ) : filtered.length === 0 ? (
          <div className="py-16 text-center">
            <Search
              size={18}
              className="text-[#4f545e] mx-auto mb-3"
            />

            <div className="text-[12px] text-[#686e79]">
              No records match your search.
            </div>

            {search && (
              <button
                onClick={() => setSearch('')}
                className="mt-3 text-[11px] text-[#76B900] hover:text-[#8bcf22]"
              >
                Clear search
              </button>
            )}
          </div>
        ) : (
          <div className="overflow-x-auto -mx-4 -mb-4">
            <table className="w-full data-table">
              <thead>
                <tr className="border-b border-[#1f2229]">
                  {COLUMNS.map((column) => (
                    <th
                      key={column.key}
                      className={`px-3 py-2.5 ${
                        column.align === 'right'
                          ? 'text-right'
                          : 'text-left'
                      }`}
                    >
                      {column.label}
                    </th>
                  ))}
                </tr>
              </thead>

              <tbody>
                {filtered.map((record) => (
                  <tr
                    key={record.record_id}
                    className="border-b border-[#1a1d24] transition-colors hover:bg-[#181b20]"
                  >
                    <td className="px-3 py-2 font-mono text-[#76B900]">
                      {record.record_id}
                    </td>

                    <td className="px-3 py-2 text-right text-[#c8cdd6]">
                      {record.laser_power_w}
                    </td>

                    <td className="px-3 py-2 text-right text-[#c8cdd6]">
                      {record.scan_speed_mm_s}
                    </td>

                    <td className="px-3 py-2 text-right text-[#c8cdd6]">
                      {record.layer_thickness_um}
                    </td>

                    <td className="px-3 py-2 text-right text-[#c8cdd6]">
                      {record.hatch_spacing_um}
                    </td>

                    <td className="px-3 py-2 text-right text-[#8b909a]">
                      {record.laser_spot_um ?? '—'}
                    </td>

                    <td className="px-3 py-2 text-right text-[#8b909a]">
                      {record.powder_size_um ?? '—'}
                    </td>

                    <td className="px-3 py-2 text-right font-medium text-[#76B900]">
                      {typeof record.ved_j_mm3 ===
                      'number'
                        ? record.ved_j_mm3.toFixed(1)
                        : '—'}
                    </td>

                    <td className="px-3 py-2 text-right text-[#c8cdd6]">
                      {record.uts_mpa ?? '—'}
                    </td>

                    <td className="px-3 py-2 text-right text-[#c8cdd6]">
                      {record.yield_strength_mpa ??
                        '—'}
                    </td>

                    <td className="px-3 py-2 text-right text-[#c8cdd6]">
                      {record.elongation_pct != null
                        ? record.elongation_pct.toFixed(1)
                        : '—'}
                    </td>

                    <td
                      className="px-3 py-2 text-[#6b7280] max-w-[160px] truncate"
                      title={record.source}
                    >
                      {record.source}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <div className="flex items-center gap-2 text-[11px] text-[#3a3d47]">
        <span>
          VED is taken from the verified processed dataset.
        </span>

        <span>·</span>

        <span>
          Record ID is a dataset identifier, not an ML
          feature.
        </span>

        <span>·</span>

        <span>
          Core process parameters: P, v, h, t.
        </span>
      </div>
    </div>
  );
}