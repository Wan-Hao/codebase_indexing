/**
 * Standard Information Retrieval evaluation metrics.
 *
 * All functions take:
 *   - rankedResults: per-query ranked list of corpus IDs (best → worst)
 *   - qrels:         ground-truth relevance: queryId → { corpusId → score }
 *   - k:             cut-off position
 *
 * Metrics implemented:
 *   - MRR@k   (Mean Reciprocal Rank)
 *   - NDCG@k  (Normalized Discounted Cumulative Gain)
 *   - Recall@k
 */

/**
 * Mean Reciprocal Rank at k.
 *
 * For each query, find the rank of the FIRST relevant document in the top-k.
 * MRR = mean(1 / rank) across all queries.
 */
export function computeMRR(
  rankedResults: Map<string, string[]>,
  qrels: Map<string, Map<string, number>>,
  k: number
): number {
  let sumRR = 0;
  let count = 0;

  for (const [queryId, ranked] of rankedResults) {
    const rels = qrels.get(queryId);
    if (!rels) continue;

    count++;
    const topK = ranked.slice(0, k);

    for (let i = 0; i < topK.length; i++) {
      const rel = rels.get(topK[i]);
      if (rel && rel > 0) {
        sumRR += 1.0 / (i + 1);
        break;
      }
    }
  }

  return count > 0 ? sumRR / count : 0;
}

/**
 * Normalized Discounted Cumulative Gain at k.
 *
 * DCG@k  = Σ (2^rel_i - 1) / log2(i + 2)   for i = 0..k-1
 * NDCG@k = DCG@k / IDCG@k
 *
 * where IDCG@k is the ideal DCG@k (ground truth sorted by relevance).
 */
export function computeNDCG(
  rankedResults: Map<string, string[]>,
  qrels: Map<string, Map<string, number>>,
  k: number
): number {
  let sumNDCG = 0;
  let count = 0;

  for (const [queryId, ranked] of rankedResults) {
    const rels = qrels.get(queryId);
    if (!rels) continue;

    count++;
    const topK = ranked.slice(0, k);

    // Compute DCG@k
    let dcg = 0;
    for (let i = 0; i < topK.length; i++) {
      const rel = rels.get(topK[i]) ?? 0;
      dcg += (Math.pow(2, rel) - 1) / Math.log2(i + 2);
    }

    // Compute IDCG@k (ideal ranking)
    const idealRels = [...rels.values()].sort((a, b) => b - a).slice(0, k);
    let idcg = 0;
    for (let i = 0; i < idealRels.length; i++) {
      idcg += (Math.pow(2, idealRels[i]) - 1) / Math.log2(i + 2);
    }

    sumNDCG += idcg > 0 ? dcg / idcg : 0;
  }

  return count > 0 ? sumNDCG / count : 0;
}

/**
 * Recall at k.
 *
 * For each query, what fraction of ALL relevant documents appear in the top-k?
 * Recall@k = |{relevant docs in top-k}| / |{all relevant docs}|
 */
export function computeRecall(
  rankedResults: Map<string, string[]>,
  qrels: Map<string, Map<string, number>>,
  k: number
): number {
  let sumRecall = 0;
  let count = 0;

  for (const [queryId, ranked] of rankedResults) {
    const rels = qrels.get(queryId);
    if (!rels) continue;

    // Count total relevant docs
    let totalRelevant = 0;
    for (const score of rels.values()) {
      if (score > 0) totalRelevant++;
    }
    if (totalRelevant === 0) continue;

    count++;
    const topK = new Set(ranked.slice(0, k));

    let hits = 0;
    for (const [corpusId, score] of rels) {
      if (score > 0 && topK.has(corpusId)) {
        hits++;
      }
    }

    sumRecall += hits / totalRelevant;
  }

  return count > 0 ? sumRecall / count : 0;
}

/**
 * Compute all standard metrics at once.
 * Returns a record of metric_name → value.
 */
export function computeAllMetrics(
  rankedResults: Map<string, string[]>,
  qrels: Map<string, Map<string, number>>
): Record<string, number> {
  return {
    'MRR@1': computeMRR(rankedResults, qrels, 1),
    'MRR@5': computeMRR(rankedResults, qrels, 5),
    'MRR@10': computeMRR(rankedResults, qrels, 10),
    'NDCG@1': computeNDCG(rankedResults, qrels, 1),
    'NDCG@5': computeNDCG(rankedResults, qrels, 5),
    'NDCG@10': computeNDCG(rankedResults, qrels, 10),
    'Recall@1': computeRecall(rankedResults, qrels, 1),
    'Recall@5': computeRecall(rankedResults, qrels, 5),
    'Recall@10': computeRecall(rankedResults, qrels, 10),
    'Recall@100': computeRecall(rankedResults, qrels, 100),
  };
}
