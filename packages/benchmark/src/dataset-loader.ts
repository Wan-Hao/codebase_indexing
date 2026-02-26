/**
 * Dataset Loader — downloads and caches CoIR benchmark datasets
 * from HuggingFace using direct Parquet file downloads.
 *
 * Strategy:
 *   1. Use /parquet API to get direct download URLs for each split
 *   2. Download parquet files (one per split, typically 0.5–30 MB)
 *   3. Convert to JSONL via pyarrow (subprocess)
 *   4. Cache JSONL for fast re-loading
 */

import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';
import type {
  BenchmarkDataset,
  BenchmarkQuery,
  CorpusEntry,
  DatasetInfo,
} from './types.js';
import { SUPPORTED_DATASETS } from './types.js';

const HF_DATASETS_API = 'https://datasets-server.huggingface.co';

export class DatasetLoader {
  private cacheDir: string;

  constructor(cacheDir = '.bench-cache') {
    this.cacheDir = cacheDir;
  }

  /**
   * Get dataset info by short ID (e.g. "codesearchnet-python").
   */
  getDatasetInfo(id: string): DatasetInfo | undefined {
    return SUPPORTED_DATASETS.find((d) => d.id === id);
  }

  /**
   * Check if a dataset is already cached locally AND non-empty.
   */
  isCached(datasetId: string): boolean {
    const dir = this.datasetDir(datasetId);
    return (
      this.fileHasContent(path.join(dir, 'corpus.jsonl')) &&
      this.fileHasContent(path.join(dir, 'queries.jsonl')) &&
      this.fileHasContent(path.join(dir, 'qrels.jsonl'))
    );
  }

  /**
   * Download a dataset from HuggingFace and cache it locally.
   * Uses parquet file direct download → pyarrow conversion.
   */
  async download(
    datasetId: string,
    onProgress?: (stage: string, detail: string) => void
  ): Promise<void> {
    const info = this.getDatasetInfo(datasetId);
    if (!info) {
      throw new Error(
        `Unknown dataset: ${datasetId}. Available: ${SUPPORTED_DATASETS.map((d) => d.id).join(', ')}`
      );
    }

    const dir = this.datasetDir(datasetId);
    fs.mkdirSync(dir, { recursive: true });

    const report = (stage: string, detail: string) => {
      onProgress?.(stage, detail);
    };

    // 1. Get parquet file URLs for both repos
    report('download', 'Fetching parquet file URLs...');
    const qcParquets = await this.fetchParquetUrls(info.hfQueriesCorpusRepo);
    const qrelsParquets = await this.fetchParquetUrls(info.hfQrelsRepo);

    // 2. Download & convert corpus
    const corpusFiles = qcParquets.filter((p) => p.split === 'corpus');
    const corpusTotalSize = corpusFiles.reduce((s, p) => s + p.size, 0);
    report('corpus', `Downloading corpus (${this.formatBytes(corpusTotalSize)})...`);
    await this.downloadAndConvertSplit(corpusFiles, path.join(dir, 'corpus.jsonl'));
    const corpusCount = this.countLines(path.join(dir, 'corpus.jsonl'));
    report('corpus', `✓ Saved ${corpusCount} corpus entries`);

    // 3. Download & convert queries
    const queryFiles = qcParquets.filter((p) => p.split === 'queries');
    const queryTotalSize = queryFiles.reduce((s, p) => s + p.size, 0);
    report('queries', `Downloading queries (${this.formatBytes(queryTotalSize)})...`);
    await this.downloadAndConvertSplit(queryFiles, path.join(dir, 'queries.jsonl'));
    const queryCount = this.countLines(path.join(dir, 'queries.jsonl'));
    report('queries', `✓ Saved ${queryCount} queries`);

    // 4. Download & convert qrels (test split)
    const qrelFiles = qrelsParquets.filter((p) => p.split === info.qrelsSplit);
    const qrelTotalSize = qrelFiles.reduce((s, p) => s + p.size, 0);
    report('qrels', `Downloading qrels (${this.formatBytes(qrelTotalSize)})...`);
    await this.downloadAndConvertSplit(qrelFiles, path.join(dir, 'qrels.jsonl'));
    const qrelCount = this.countLines(path.join(dir, 'qrels.jsonl'));
    report('qrels', `✓ Saved ${qrelCount} relevance judgments`);

    // Validate
    if (corpusCount === 0) throw new Error('Downloaded 0 corpus entries');
    if (queryCount === 0) throw new Error('Downloaded 0 queries');
    if (qrelCount === 0) throw new Error('Downloaded 0 qrels');

    // Save metadata
    fs.writeFileSync(
      path.join(dir, 'meta.json'),
      JSON.stringify(
        {
          datasetId,
          hfQueriesCorpusRepo: info.hfQueriesCorpusRepo,
          hfQrelsRepo: info.hfQrelsRepo,
          qrelsSplit: info.qrelsSplit,
          corpusCount,
          queryCount,
          qrelCount,
          downloadedAt: new Date().toISOString(),
        },
        null,
        2
      )
    );
  }

  /**
   * Load a dataset from the local cache. Downloads if not cached.
   */
  async load(
    datasetId: string,
    opts?: { maxCorpus?: number; maxQueries?: number },
    onProgress?: (stage: string, detail: string) => void
  ): Promise<BenchmarkDataset> {
    if (!this.isCached(datasetId)) {
      await this.download(datasetId, onProgress);
    }

    const dir = this.datasetDir(datasetId);
    onProgress?.('load', 'Loading dataset from cache...');

    // Load corpus
    const corpusRaw = this.loadJsonl(path.join(dir, 'corpus.jsonl'));

    // Load qrels FIRST so we can do smart corpus subsetting
    const qrelsRaw = this.loadJsonl(path.join(dir, 'qrels.jsonl'));
    const qrels = new Map<string, Map<string, number>>();
    const relevantCorpusIds = new Set<string>();
    for (const row of qrelsRaw) {
      const qid = String(row['query_id'] ?? row['query-id'] ?? '');
      const cid = String(row['corpus_id'] ?? row['corpus-id'] ?? '');
      const score = Number(row['score'] ?? 1);
      if (!qrels.has(qid)) {
        qrels.set(qid, new Map());
      }
      qrels.get(qid)!.set(cid, score);
      if (score > 0) {
        relevantCorpusIds.add(cid);
      }
    }

    // Build corpus: when maxCorpus is set, ensure relevant documents are included
    let corpus: CorpusEntry[];
    if (opts?.maxCorpus && opts.maxCorpus < corpusRaw.length) {
      const corpusMap = new Map<string, CorpusEntry>();
      // First, add all relevant documents (ground truth)
      for (const row of corpusRaw) {
        const id = String(row['_id']);
        if (relevantCorpusIds.has(id)) {
          corpusMap.set(id, {
            id,
            text: String(row['text'] ?? ''),
            title: row['title'] ? String(row['title']) : undefined,
          });
        }
      }
      // Then fill up with non-relevant documents as distractors
      for (const row of corpusRaw) {
        if (corpusMap.size >= opts.maxCorpus) break;
        const id = String(row['_id']);
        if (!corpusMap.has(id)) {
          corpusMap.set(id, {
            id,
            text: String(row['text'] ?? ''),
            title: row['title'] ? String(row['title']) : undefined,
          });
        }
      }
      corpus = [...corpusMap.values()];
    } else {
      corpus = corpusRaw.map((row) => ({
        id: String(row['_id']),
        text: String(row['text'] ?? ''),
        title: row['title'] ? String(row['title']) : undefined,
      }));
    }

    // Load queries
    const queriesRaw = this.loadJsonl(path.join(dir, 'queries.jsonl'));
    const queries: BenchmarkQuery[] = queriesRaw.map((row) => ({
      id: String(row['_id']),
      text: String(row['text'] ?? ''),
    }));

    // Filter queries to only those with qrels + at least one relevant doc in corpus
    const corpusIds = new Set(corpus.map((c) => c.id));
    const filteredQueries = queries.filter((q) => {
      const rels = qrels.get(q.id);
      if (!rels) return false;
      for (const cid of rels.keys()) {
        if (corpusIds.has(cid)) return true;
      }
      return false;
    });

    // Apply max queries AFTER filtering
    let finalQueries = filteredQueries;
    if (opts?.maxQueries && opts.maxQueries < finalQueries.length) {
      finalQueries = finalQueries.slice(0, opts.maxQueries);
    }

    onProgress?.(
      'load',
      `Loaded ${finalQueries.length} queries, ${corpus.length} corpus entries, ${qrels.size} qrel groups`
    );

    return {
      name: datasetId,
      queries: finalQueries,
      corpus,
      qrels,
    };
  }

  // ---- Parquet download + conversion helpers ----

  /**
   * Get parquet file URLs for a dataset from HuggingFace Datasets Server.
   */
  private async fetchParquetUrls(
    hfRepo: string
  ): Promise<Array<{ split: string; url: string; size: number; filename: string }>> {
    const url = `${HF_DATASETS_API}/parquet?dataset=${encodeURIComponent(hfRepo)}`;
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Failed to fetch parquet URLs for ${hfRepo}: ${resp.status} ${resp.statusText}`);
    }
    const data = (await resp.json()) as {
      parquet_files: Array<{
        split: string;
        url: string;
        size: number;
        filename: string;
      }>;
    };
    return data.parquet_files;
  }

  /**
   * Download parquet file(s) and convert to JSONL using pyarrow.
   * Handles multiple shards per split.
   */
  private async downloadAndConvertSplit(
    files: Array<{ url: string; size: number; filename: string }>,
    outputJsonlPath: string
  ): Promise<void> {
    // Create temp directory for parquet files
    const tmpDir = path.join(path.dirname(outputJsonlPath), '_tmp_parquet');
    fs.mkdirSync(tmpDir, { recursive: true });

    try {
      // Download each parquet file
      const localPaths: string[] = [];
      for (const file of files) {
        const localPath = path.join(tmpDir, file.filename);
        const resp = await fetch(file.url);
        if (!resp.ok) {
          throw new Error(`Failed to download parquet: ${resp.status} ${resp.statusText} (${file.url})`);
        }
        const buffer = Buffer.from(await resp.arrayBuffer());
        fs.writeFileSync(localPath, buffer);
        localPaths.push(localPath);
      }

      // Convert all parquet files to a single JSONL using pyarrow
      const pathsJson = JSON.stringify(localPaths);
      const pyScriptPath = path.join(tmpDir, '_convert.py');
      fs.writeFileSync(
        pyScriptPath,
        [
          'import json, sys',
          'import pyarrow.parquet as pq',
          '',
          'paths = json.loads(sys.argv[1])',
          'output = sys.argv[2]',
          '',
          'with open(output, "w") as f:',
          '    for p in paths:',
          '        table = pq.read_table(p)',
          '        for batch in table.to_batches():',
          '            for row in batch.to_pylist():',
          '                f.write(json.dumps(row, ensure_ascii=False) + "\\n")',
        ].join('\n'),
        'utf-8'
      );
      execSync(
        `python3 ${JSON.stringify(pyScriptPath)} ${JSON.stringify(pathsJson)} ${JSON.stringify(outputJsonlPath)}`,
        { stdio: ['pipe', 'pipe', 'pipe'], maxBuffer: 50 * 1024 * 1024 }
      );
    } finally {
      // Clean up temp parquet files
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  }

  // ---- Local file helpers ----

  private datasetDir(datasetId: string): string {
    return path.resolve(this.cacheDir, 'datasets', datasetId);
  }

  private fileHasContent(filePath: string): boolean {
    try {
      const stat = fs.statSync(filePath);
      return stat.size > 10;
    } catch {
      return false;
    }
  }

  private countLines(filePath: string): number {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      return content.split('\n').filter((l) => l.trim().length > 0).length;
    } catch {
      return 0;
    }
  }

  private loadJsonl(filePath: string): Record<string, unknown>[] {
    const content = fs.readFileSync(filePath, 'utf-8');
    return content
      .split('\n')
      .filter((line) => line.trim().length > 0)
      .map((line) => JSON.parse(line) as Record<string, unknown>);
  }

  private formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
}
