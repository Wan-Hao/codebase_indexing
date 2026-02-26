/**
 * Type definitions for the benchmark evaluation system.
 *
 * Follows the CoIR dataset format on HuggingFace:
 *   - Repo "{name}-queries-corpus": config "default", splits "corpus" & "queries"
 *     columns: _id, text, title, partition, language, ...
 *   - Repo "{name}-qrels": config "default", splits "train" / "test" / "valid"
 *     columns: query_id, corpus_id, score
 */

/** A natural-language query in the benchmark dataset */
export interface BenchmarkQuery {
  id: string;
  text: string;
}

/** A code snippet in the benchmark corpus */
export interface CorpusEntry {
  id: string;
  text: string;
  title?: string;
}

/**
 * A fully loaded benchmark dataset ready for evaluation.
 */
export interface BenchmarkDataset {
  /** Dataset identifier (e.g. "codesearchnet-python") */
  name: string;
  /** Natural-language queries */
  queries: BenchmarkQuery[];
  /** Code corpus entries */
  corpus: CorpusEntry[];
  /**
   * Relevance judgments: queryId → { corpusId → relevance_score }
   * Score is typically 1 (relevant) or 0 (not relevant).
   */
  qrels: Map<string, Map<string, number>>;
}

/** Standard IR evaluation metrics at various cut-offs */
export interface EvalMetrics {
  mrr_at_1: number;
  mrr_at_5: number;
  mrr_at_10: number;
  ndcg_at_1: number;
  ndcg_at_5: number;
  ndcg_at_10: number;
  recall_at_1: number;
  recall_at_5: number;
  recall_at_10: number;
  recall_at_100: number;
}

/** Complete benchmark evaluation result */
export interface BenchmarkResult {
  /** Dataset name */
  dataset: string;
  /** Embedding model used */
  model: string;
  /** Evaluation metrics */
  metrics: EvalMetrics;
  /** Number of queries evaluated */
  numQueries: number;
  /** Number of corpus entries */
  numCorpus: number;
  /** Timing breakdown */
  timing: {
    embedCorpusMs: number;
    embedQueriesMs: number;
    searchMs: number;
    totalMs: number;
  };
}

/** Options for running a benchmark */
export interface RunOptions {
  /** Embedding model name (default: Xenova/bge-base-en-v1.5) */
  model?: string;
  /** Maximum number of corpus entries (for quick testing) */
  maxCorpus?: number;
  /** Maximum number of queries (for quick testing) */
  maxQueries?: number;
  /** Batch size for embedding */
  batchSize?: number;
  /** Directory for caching downloaded datasets and embeddings */
  cacheDir?: string;
}

/** Metadata for a supported benchmark dataset */
export interface DatasetInfo {
  /** Short identifier */
  id: string;
  /** HuggingFace repo for queries + corpus */
  hfQueriesCorpusRepo: string;
  /** HuggingFace repo for qrels (relevance judgments) */
  hfQrelsRepo: string;
  /** Which split of qrels to use for evaluation */
  qrelsSplit: string;
  /** Human-readable description */
  description: string;
  /** Programming language */
  language: string;
}

/**
 * Registry of supported benchmark datasets.
 * All follow the CoIR format on HuggingFace (two repos per dataset).
 */
export const SUPPORTED_DATASETS: DatasetInfo[] = [
  {
    id: 'codesearchnet-python',
    hfQueriesCorpusRepo: 'CoIR-Retrieval/CodeSearchNet-python-queries-corpus',
    hfQrelsRepo: 'CoIR-Retrieval/CodeSearchNet-python-qrels',
    qrelsSplit: 'test',
    description: 'CodeSearchNet Python — NL query → Python function retrieval (14.9K test queries, 280K corpus)',
    language: 'python',
  },
  {
    id: 'codesearchnet-javascript',
    hfQueriesCorpusRepo: 'CoIR-Retrieval/CodeSearchNet-javascript-queries-corpus',
    hfQrelsRepo: 'CoIR-Retrieval/CodeSearchNet-javascript-qrels',
    qrelsSplit: 'test',
    description: 'CodeSearchNet JavaScript — NL query → JS function retrieval',
    language: 'javascript',
  },
  {
    id: 'codesearchnet-go',
    hfQueriesCorpusRepo: 'CoIR-Retrieval/CodeSearchNet-go-queries-corpus',
    hfQrelsRepo: 'CoIR-Retrieval/CodeSearchNet-go-qrels',
    qrelsSplit: 'test',
    description: 'CodeSearchNet Go — NL query → Go function retrieval',
    language: 'go',
  },
  {
    id: 'codesearchnet-java',
    hfQueriesCorpusRepo: 'CoIR-Retrieval/CodeSearchNet-java-queries-corpus',
    hfQrelsRepo: 'CoIR-Retrieval/CodeSearchNet-java-qrels',
    qrelsSplit: 'test',
    description: 'CodeSearchNet Java — NL query → Java method retrieval',
    language: 'java',
  },
  {
    id: 'codesearchnet-ruby',
    hfQueriesCorpusRepo: 'CoIR-Retrieval/CodeSearchNet-ruby-queries-corpus',
    hfQrelsRepo: 'CoIR-Retrieval/CodeSearchNet-ruby-qrels',
    qrelsSplit: 'test',
    description: 'CodeSearchNet Ruby — NL query → Ruby method retrieval',
    language: 'ruby',
  },
  {
    id: 'codesearchnet-php',
    hfQueriesCorpusRepo: 'CoIR-Retrieval/CodeSearchNet-php-queries-corpus',
    hfQrelsRepo: 'CoIR-Retrieval/CodeSearchNet-php-qrels',
    qrelsSplit: 'test',
    description: 'CodeSearchNet PHP — NL query → PHP function retrieval',
    language: 'php',
  },
  {
    id: 'cosqa',
    hfQueriesCorpusRepo: 'CoIR-Retrieval/cosqa-queries-corpus',
    hfQrelsRepo: 'CoIR-Retrieval/cosqa-qrels',
    qrelsSplit: 'test',
    description: 'CoSQA — Web search queries → Python code retrieval (500 test queries, 20K corpus)',
    language: 'python',
  },
];
