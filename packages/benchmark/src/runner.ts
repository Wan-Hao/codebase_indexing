/**
 * Benchmark Runner — orchestrates the full evaluation pipeline:
 *
 * 1. Load dataset (corpus + queries + qrels)
 * 2. Embed corpus entries  → Float32Array matrix [n_corpus × dim]
 * 3. Embed queries          → Float32Array matrix [n_queries × dim]
 * 4. For each query, brute-force cosine similarity → ranked list
 * 5. Compute IR metrics (MRR, NDCG, Recall)
 *
 * Notes:
 * - Uses brute-force search (not ANN) for reproducible, exact evaluation.
 * - Embeddings are normalized, so cosine_sim = dot_product.
 * - Caches embedding matrices to disk for fast re-runs.
 */

import fs from 'node:fs';
import path from 'node:path';
import { Embedder } from '@codebase-indexer/core';
import { DatasetLoader } from './dataset-loader.js';
import { computeAllMetrics } from './metrics.js';
import type {
  BenchmarkDataset,
  BenchmarkResult,
  EvalMetrics,
  RunOptions,
} from './types.js';

/** Sanitize model name for use in file paths */
function modelToPathSafe(model: string): string {
  return model.replace(/[\/\\:]/g, '_');
}

export class BenchmarkRunner {
  private loader: DatasetLoader;
  private cacheDir: string;

  constructor(cacheDir = '.bench-cache') {
    this.cacheDir = cacheDir;
    this.loader = new DatasetLoader(cacheDir);
  }

  /**
   * Run a full benchmark evaluation.
   */
  async run(
    datasetId: string,
    opts: RunOptions = {},
    onProgress?: (stage: string, detail: string) => void
  ): Promise<BenchmarkResult> {
    const totalStart = Date.now();
    const report = (stage: string, detail: string) => {
      onProgress?.(stage, detail);
    };

    const model = opts.model ?? 'Xenova/bge-m3';
    const batchSize = opts.batchSize ?? 32;

    // Step 1: Load dataset
    report('dataset', 'Loading dataset...');
    const dataset = await this.loader.load(
      datasetId,
      { maxCorpus: opts.maxCorpus, maxQueries: opts.maxQueries },
      onProgress
    );
    report('dataset', `Loaded ${dataset.queries.length} queries, ${dataset.corpus.length} corpus entries`);

    // Step 2: Initialize embedder
    report('model', `Loading embedding model: ${model}...`);
    const embedder = new Embedder(model);
    await embedder.init();
    report('model', 'Model loaded');

    const dim = embedder.getDimension();

    // Step 3: Embed corpus
    report('embed-corpus', `Embedding ${dataset.corpus.length} corpus entries...`);
    const corpusStart = Date.now();
    const corpusEmbeddings = await this.embedWithCache(
      embedder,
      dataset.corpus.map((c) => c.text),
      dataset.name,
      'corpus',
      model,
      batchSize,
      (done, total) => report('embed-corpus', `Embedding corpus: ${done}/${total}`)
    );
    const corpusMs = Date.now() - corpusStart;
    report('embed-corpus', `✓ Corpus embedded in ${(corpusMs / 1000).toFixed(1)}s`);

    // Step 4: Embed queries
    report('embed-queries', `Embedding ${dataset.queries.length} queries...`);
    const queriesStart = Date.now();
    const queryEmbeddings = await this.embedWithCache(
      embedder,
      dataset.queries.map((q) => q.text),
      dataset.name,
      'queries',
      model,
      batchSize,
      (done, total) => report('embed-queries', `Embedding queries: ${done}/${total}`)
    );
    const queriesMs = Date.now() - queriesStart;
    report('embed-queries', `✓ Queries embedded in ${(queriesMs / 1000).toFixed(1)}s`);

    // Step 5: Brute-force retrieval
    report('search', 'Computing similarities and ranking...');
    const searchStart = Date.now();
    const rankedResults = this.bruteForceSearch(
      dataset,
      queryEmbeddings,
      corpusEmbeddings,
      dim,
      (done, total) => report('search', `Ranking queries: ${done}/${total}`)
    );
    const searchMs = Date.now() - searchStart;
    report('search', `✓ Search completed in ${(searchMs / 1000).toFixed(1)}s`);

    // Step 6: Compute metrics
    report('metrics', 'Computing evaluation metrics...');
    const metricsRaw = computeAllMetrics(rankedResults, dataset.qrels);

    const metrics: EvalMetrics = {
      mrr_at_1: metricsRaw['MRR@1'],
      mrr_at_5: metricsRaw['MRR@5'],
      mrr_at_10: metricsRaw['MRR@10'],
      ndcg_at_1: metricsRaw['NDCG@1'],
      ndcg_at_5: metricsRaw['NDCG@5'],
      ndcg_at_10: metricsRaw['NDCG@10'],
      recall_at_1: metricsRaw['Recall@1'],
      recall_at_5: metricsRaw['Recall@5'],
      recall_at_10: metricsRaw['Recall@10'],
      recall_at_100: metricsRaw['Recall@100'],
    };

    const totalMs = Date.now() - totalStart;
    report('done', `Benchmark completed in ${(totalMs / 1000).toFixed(1)}s`);

    return {
      dataset: datasetId,
      model,
      metrics,
      numQueries: dataset.queries.length,
      numCorpus: dataset.corpus.length,
      timing: {
        embedCorpusMs: corpusMs,
        embedQueriesMs: queriesMs,
        searchMs,
        totalMs,
      },
    };
  }

  // ---- Embedding with disk cache ----

  /**
   * Embed a list of texts, caching the result as a binary Float32Array file.
   * On cache hit, loads from disk instead of recomputing.
   */
  private async embedWithCache(
    embedder: Embedder,
    texts: string[],
    datasetId: string,
    splitName: string,
    model: string,
    batchSize: number,
    onProgress?: (done: number, total: number) => void
  ): Promise<Float32Array> {
    const cacheKey = `${datasetId}/${modelToPathSafe(model)}`;
    const cachePath = path.resolve(
      this.cacheDir,
      'embeddings',
      cacheKey,
      `${splitName}-${texts.length}.bin`
    );

    // Check cache
    if (fs.existsSync(cachePath)) {
      const buffer = fs.readFileSync(cachePath);
      return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    }

    // Embed in batches
    const dim = embedder.getDimension();
    const result = new Float32Array(texts.length * dim);

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const embeddings = await embedder.embedBatch(batch, batchSize);

      for (let j = 0; j < embeddings.length; j++) {
        result.set(embeddings[j], (i + j) * dim);
      }

      onProgress?.(Math.min(i + batchSize, texts.length), texts.length);
    }

    // Save to cache
    const dir = path.dirname(cachePath);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(cachePath, Buffer.from(result.buffer));

    return result;
  }

  // ---- Brute-force cosine similarity search ----

  /**
   * For each query, compute dot-product similarity with all corpus entries,
   * return a Map of queryId → ranked list of corpusIds.
   *
   * Since embeddings are L2-normalized, dot product = cosine similarity.
   */
  private bruteForceSearch(
    dataset: BenchmarkDataset,
    queryEmbeddings: Float32Array,
    corpusEmbeddings: Float32Array,
    dim: number,
    onProgress?: (done: number, total: number) => void
  ): Map<string, string[]> {
    const results = new Map<string, string[]>();
    const nCorpus = dataset.corpus.length;
    const nQueries = dataset.queries.length;

    // Max results to track per query (for Recall@100 we need at least 100)
    const maxK = 100;

    for (let qi = 0; qi < nQueries; qi++) {
      const queryId = dataset.queries[qi].id;
      const qOffset = qi * dim;

      // Compute similarities with all corpus entries
      // Use a partial sort (find top-K) for efficiency
      const scores: Array<{ idx: number; score: number }> = new Array(nCorpus);
      for (let ci = 0; ci < nCorpus; ci++) {
        const cOffset = ci * dim;
        let dot = 0;
        for (let d = 0; d < dim; d++) {
          dot += queryEmbeddings[qOffset + d] * corpusEmbeddings[cOffset + d];
        }
        scores[ci] = { idx: ci, score: dot };
      }

      // Partial sort: only need top-maxK
      scores.sort((a, b) => b.score - a.score);

      const ranked = scores
        .slice(0, maxK)
        .map((s) => dataset.corpus[s.idx].id);

      results.set(queryId, ranked);

      if ((qi + 1) % 100 === 0 || qi === nQueries - 1) {
        onProgress?.(qi + 1, nQueries);
      }
    }

    return results;
  }
}
