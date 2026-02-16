/**
 * Shared type definitions for the codebase indexer.
 */

/** A code chunk extracted from AST-based splitting */
export interface CodeChunk {
  /** Unique identifier based on content hash */
  chunkId: string;
  /** Relative file path */
  filePath: string;
  /** Start line number (1-based) */
  startLine: number;
  /** End line number (1-based, inclusive) */
  endLine: number;
  /** The actual code content */
  content: string;
  /** SHA-256 hash of the content */
  contentHash: string;
  /** AST node type (e.g., "function_declaration", "class_declaration") */
  nodeType: string;
  /** Name of the symbol if available (function name, class name, etc.) */
  symbolName?: string;
}

/** File hash entry for Merkle tree construction */
export interface FileHashEntry {
  path: string;
  hash: string;
}

/** Merkle tree node */
export interface MerkleNode {
  path: string;
  hash: string;
  isFile: boolean;
  children: string[];
}

/** Result of diffing two Merkle trees */
export interface MerkleDiff {
  added: string[];
  removed: string[];
  modified: string[];
}

/** A search result returned from vector search */
export interface SearchResult {
  /** Relative file path */
  filePath: string;
  /** Start line */
  startLine: number;
  /** End line */
  endLine: number;
  /** Similarity score (0-1) */
  score: number;
  /** The actual code content (read from local disk) */
  content: string;
  /** AST node type */
  nodeType: string;
  /** Symbol name if available */
  symbolName?: string;
}

/** Indexing statistics */
export interface IndexStats {
  totalFiles: number;
  totalChunks: number;
  newChunks: number;
  cachedChunks: number;
  indexTimeMs: number;
}

/** Configuration for the indexer */
export interface IndexerConfig {
  /** Root directory to index */
  rootDir: string;
  /** File extensions to include (e.g., [".ts", ".tsx"]) */
  extensions: string[];
  /** Qdrant server URL */
  qdrantUrl: string;
  /** Qdrant collection name */
  collectionName: string;
  /** Embedding model name */
  embeddingModel: string;
  /** Maximum tokens per chunk */
  maxChunkTokens: number;
  /** Minimum tokens per chunk (smaller chunks get merged with siblings) */
  minChunkTokens: number;
  /** Path for the local embedding cache */
  cachePath: string;
  /** Number of top results to return in search */
  topK: number;
}

/** Default configuration */
export const DEFAULT_CONFIG: Partial<IndexerConfig> = {
  extensions: ['.ts', '.tsx'],
  qdrantUrl: 'http://localhost:6333',
  collectionName: 'codebase',
  embeddingModel: 'Xenova/bge-base-en-v1.5',
  maxChunkTokens: 512,
  minChunkTokens: 30,
  cachePath: '.cache/embeddings',
  topK: 10,
};
