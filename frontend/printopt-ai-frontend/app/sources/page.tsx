'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  Database,
  ExternalLink,
  FileText,
  FlaskConical,
  Search,
  RefreshCw,
} from 'lucide-react';

import { PageHeader } from '@/components/layout/PageHeader';
import { Card } from '@/components/ui/Card';
import { StatusBanner } from '@/components/layout/StatusBanner';
import { Badge } from '@/components/ui/Badge';

interface SourceRecord {
  id: string;
  title: string;
  authors: string[];
  year: number | null;
  doi: string | null;
  source_type: string;
  record_count: number;
  journal: string | null;
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ??
  'http://127.0.0.1:8000';

function isLabSource(source: SourceRecord) {
  return source.id === 'LAB';
}

function getReferenceLabel(source: SourceRecord) {
  if (isLabSource(source)) {
    return 'Laboratory';
  }

  return source.id;
}

function getSourceTypeLabel(source: SourceRecord) {
  if (isLabSource(source)) {
    return 'Experimental';
  }

  if (source.source_type && source.source_type !== 'other') {
    return source.source_type;
  }

  return 'Literature';
}

function getDoiUrl(doi: string) {
  if (doi.startsWith('http://') || doi.startsWith('https://')) {
    return doi;
  }

  return `https://doi.org/${doi}`;
}

export default function SourcesPage() {
  const [sources, setSources] = useState<SourceRecord[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadSources() {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE}/api/sources`,
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
        (await response.json()) as SourceRecord[];

      if (!Array.isArray(data)) {
        throw new Error(
          'Invalid source data received from API.'
        );
      }

      setSources(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to load source data.'
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadSources();
  }, []);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) {
      return sources;
    }

    return sources.filter((source) => {
      const searchableText = [
        source.id,
        source.title,
        source.source_type,
        source.journal ?? '',
        source.year?.toString() ?? '',
        ...source.authors,
      ]
        .join(' ')
        .toLowerCase();

      return searchableText.includes(query);
    });
  }, [sources, search]);

  const totalRecords = useMemo(() => {
    return sources.reduce(
      (total, source) =>
        total + (source.record_count ?? 0),
      0
    );
  }, [sources]);

  const laboratorySources = useMemo(() => {
    return sources.filter(isLabSource).length;
  }, [sources]);

  const literatureSources = sources.length - laboratorySources;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Sources"
        subtitle="Research provenance for the experimental dataset"
        badge={
          loading
            ? 'Loading Sources'
            : `${sources.length} Sources`
        }
        actions={
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search
                size={12}
                className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[#5a5f6b]"
              />

              <input
                className="pl-7 pr-3 py-1.5 text-[12px] bg-[#16181c] border border-[#2a2d35] rounded text-[#c8cdd6] placeholder-[#3a3d47] focus:outline-none focus:border-[#76B900]/50 w-56"
                placeholder="Search sources..."
                value={search}
                onChange={(e) =>
                  setSearch(e.target.value)
                }
              />
            </div>

            <button
              onClick={loadSources}
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
          </div>
        }
      />

      {error && (
        <StatusBanner
          variant="warning"
          title="Source data unavailable"
          message={error}
        />
      )}

      {!loading && !error && (
        <StatusBanner
          variant="pending"
          title="Dataset provenance"
          message="Sources shown here are loaded directly from the experimental dataset. Citation metadata is displayed only where it is provided by the dataset."
        />
      )}

      <div className="grid grid-cols-4 gap-4">
        <Card
          title="Total Sources"
          subtitle="Unique source records"
        >
          <div className="flex items-center gap-2">
            <FileText
              size={16}
              className="text-[#76b900]"
            />

            <span className="text-[20px] font-semibold text-[#f0f1f3] font-mono">
              {loading ? '—' : sources.length}
            </span>
          </div>
        </Card>

        <Card
          title="Experimental Records"
          subtitle="Records represented by sources"
        >
          <div className="flex items-center gap-2">
            <Database
              size={16}
              className="text-[#76b900]"
            />

            <span className="text-[20px] font-semibold text-[#f0f1f3] font-mono">
              {loading ? '—' : totalRecords}
            </span>
          </div>
        </Card>

        <Card
          title="Literature Sources"
          subtitle="Referenced publications and theses"
        >
          <div className="text-[20px] font-semibold text-[#f0f1f3] font-mono">
            {loading ? '—' : literatureSources}
          </div>
        </Card>

        <Card
          title="Laboratory Data"
          subtitle="Local experimental source"
        >
          <div className="flex items-center gap-2">
            <FlaskConical
              size={16}
              className="text-[#76b900]"
            />

            <span className="text-[20px] font-semibold text-[#f0f1f3] font-mono">
              {loading
                ? '—'
                : laboratorySources}
            </span>

            {!loading && laboratorySources > 0 && (
              <span className="text-[11px] text-[#7f8490]">
                source
              </span>
            )}
          </div>
        </Card>
      </div>

      <Card
        title="Research Sources"
        subtitle={
          search
            ? `${filtered.length} matching sources`
            : 'References contributing experimental records to the dataset'
        }
      >
        {loading ? (
          <div className="py-16 text-center">
            <RefreshCw
              size={18}
              className="animate-spin text-[#76b900] mx-auto mb-3"
            />

            <div className="text-[12px] text-[#686e79]">
              Loading source records...
            </div>
          </div>
        ) : error ? (
          <div className="py-16 text-center">
            <div className="text-[12px] text-[#686e79]">
              Source records could not be loaded.
            </div>
          </div>
        ) : filtered.length === 0 ? (
          <div className="py-16 text-center">
            <Search
              size={18}
              className="text-[#4f545e] mx-auto mb-3"
            />

            <div className="text-[12px] text-[#686e79]">
              No sources match your search.
            </div>

            {search && (
              <button
                onClick={() => setSearch('')}
                className="mt-3 text-[11px] text-[#76b900] hover:text-[#8bcf22]"
              >
                Clear search
              </button>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            {filtered.map((source) => (
              <div
                key={source.id}
                className="bg-[#16181c] border border-[#1f2229] hover:border-[#2a2d35] rounded p-4 transition-colors"
              >
                <div className="flex items-start gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start gap-3 mb-2">
                      <div className="shrink-0">
                        <span className="inline-flex h-7 min-w-7 px-2 items-center justify-center rounded bg-[#20242b] border border-[#2b2f37] text-[9px] font-semibold text-[#9ca1ac]">
                          {getReferenceLabel(
                            source
                          )}
                        </span>
                      </div>

                      <h3 className="text-[13px] font-medium text-[#c8cdd6] leading-relaxed">
                        {source.title}
                      </h3>
                    </div>

                    {source.authors.length > 0 && (
                      <div className="text-[11px] text-[#666b76] ml-10">
                        {source.authors
                          .slice(0, 4)
                          .join(', ')}

                        {source.authors.length > 4 &&
                          ' et al.'}
                      </div>
                    )}

                    {(source.journal ||
                      source.year) && (
                      <div className="flex items-center gap-2 flex-wrap ml-10 mt-1">
                        {source.journal && (
                          <span className="text-[11px] italic text-[#5b606b]">
                            {source.journal}
                          </span>
                        )}

                        {source.year && (
                          <span className="text-[11px] text-[#454a54]">
                            ({source.year})
                          </span>
                        )}
                      </div>
                    )}

                    {source.doi && (
                      <a
                        href={getDoiUrl(
                          source.doi
                        )}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-[11px] text-[#76b900] hover:text-[#8bcf22] mt-2 ml-10 transition-colors"
                      >
                        <span>
                          DOI: {source.doi}
                        </span>

                        <ExternalLink
                          size={9}
                        />
                      </a>
                    )}
                  </div>

                  <div className="flex flex-col items-end gap-2 shrink-0">
                    <Badge variant="gray">
                      {getSourceTypeLabel(
                        source
                      )}
                    </Badge>

                    <div className="text-[11px] text-[#5a5f6b]">
                      <span className="font-semibold text-[#76B900]">
                        {source.record_count}
                      </span>

                      <span className="text-[#3a3d47]">
                        {' '}
                        records
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}