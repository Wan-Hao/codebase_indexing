/**
 * Core Indexer: Orchestrates the full indexing pipeline.
 *
 * Flow:
 * 1. Scan files (Rust native scanner)
 * 2. Compute file hashes (Rust SHA-256)
 * 3. Build Merkle tree (Rust)
 * 4. Diff with previous Merkle tree to find changed files
 * 5. Chunk changed files (tree-sitter AST)
 * 6. Check embedding cache (content-hash)
 * 7. Generate embeddings for uncached chunks (HuggingFace transformers)
 * 8. Upsert into Qdrant
 * 9. Save updated Merkle tree and cache
 */

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import ignore from 'ignore';
import type {
  CodeChunk,
  FileHashEntry,
  IndexerConfig,
  IndexStats,
  MerkleNode,
} from './types.js';
import { ASTChunker } from './chunker/ast-chunker.js';
import { Embedder } from './embedding/embedder.js';
import { VectorStore } from './storage/vector-store.js';
import { EmbeddingCache } from './storage/cache.js';

// Native bindings (will be available after `napi build`)
// For development, we provide TS fallbacks
let native: {
  scanDirectory?: (rootPath: string, extensions: string[]) => string[];
  sha256HashFiles?: (filePaths: string[]) => Array<{ path: string; hash: string }>;
  buildMerkleTree?: (fileHashes: FileHashEntry[]) => MerkleNode[];
  diffMerkleTrees?: (oldNodes: MerkleNode[], newNodes: MerkleNode[]) => { added: string[]; removed: string[]; modified: string[] };
  getRootHash?: (nodes: MerkleNode[]) => string | null;
} | null = null;

async function loadNative() {
  try {
    native = await import('../native/index.js');
  } catch {
    console.warn(
      '⚠ Native module not found. Using TypeScript fallbacks. Run `npm run build:rs` to build.'
    );
  }
}

// ---- TS Fallbacks for native functions ----

function tsSha256HashFile(filePath: string): string {
  const content = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(content).digest('hex');
}

function tsSha256HashFiles(
  filePaths: string[]
): Array<{ path: string; hash: string }> {
  return filePaths
    .map((p) => {
      try {
        return { path: p, hash: tsSha256HashFile(p) };
      } catch {
        return null;
      }
    })
    .filter((x): x is { path: string; hash: string } => x !== null);
}

function tsBuildMerkleTree(
  fileHashes: FileHashEntry[]
): MerkleNode[] {
  const nodes = new Map<string, MerkleNode>();
  const dirChildren = new Map<string, Set<string>>();

  // Insert leaf nodes
  for (const fh of fileHashes) {
    nodes.set(fh.path, {
      path: fh.path,
      hash: fh.hash,
      isFile: true,
      children: [],
    });

    // Register parent directories
    let current = fh.path;
    while (true) {
      const parent = tsParentPath(current);
      if (!dirChildren.has(parent)) {
        dirChildren.set(parent, new Set());
      }
      dirChildren.get(parent)!.add(current);
      if (parent === current || parent === '.') {
        if (parent === '.') {
          dirChildren.get('.')!.add(current);
        }
        break;
      }
      current = parent;
    }
  }

  // Build directory nodes bottom-up
  const dirPaths = [...dirChildren.keys()].sort(
    (a, b) => b.split('/').length - a.split('/').length
  );

  for (const dirPath of dirPaths) {
    const children = [...(dirChildren.get(dirPath) ?? [])].sort();
    const childHashes = children
      .map((c) => nodes.get(c)?.hash ?? '')
      .sort()
      .join('');

    const hash = crypto
      .createHash('sha256')
      .update(childHashes)
      .digest('hex');

    nodes.set(dirPath, {
      path: dirPath,
      hash,
      isFile: false,
      children,
    });
  }

  return [...nodes.values()];
}

function tsDiffMerkleTrees(
  oldNodes: MerkleNode[],
  newNodes: MerkleNode[]
): { added: string[]; removed: string[]; modified: string[] } {
  const oldFiles = new Map(
    oldNodes.filter((n) => n.isFile).map((n) => [n.path, n.hash])
  );
  const newFiles = new Map(
    newNodes.filter((n) => n.isFile).map((n) => [n.path, n.hash])
  );

  const added: string[] = [];
  const removed: string[] = [];
  const modified: string[] = [];

  for (const [p, hash] of newFiles) {
    if (!oldFiles.has(p)) {
      added.push(p);
    } else if (oldFiles.get(p) !== hash) {
      modified.push(p);
    }
  }

  for (const p of oldFiles.keys()) {
    if (!newFiles.has(p)) {
      removed.push(p);
    }
  }

  return { added, removed, modified };
}

function tsParentPath(p: string): string {
  const idx = p.lastIndexOf('/');
  return idx >= 0 ? p.substring(0, idx) : '.';
}

// ---- Scanner fallback ----

function tsScanDirectory(rootPath: string, extensions: string[]): string[] {
  const results: string[] = [];
  const extSet = new Set(extensions.map((e) => e.toLowerCase()));

  // Load .gitignore / .cursorignore
  let ig = ignore();
  const gitignorePath = path.join(rootPath, '.gitignore');
  if (fs.existsSync(gitignorePath)) {
    ig = ig.add(fs.readFileSync(gitignorePath, 'utf-8'));
  }
  const cursorignorePath = path.join(rootPath, '.cursorignore');
  if (fs.existsSync(cursorignorePath)) {
    ig = ig.add(fs.readFileSync(cursorignorePath, 'utf-8'));
  }

  function walk(dir: string) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue;
      const fullPath = path.join(dir, entry.name);
      const relPath = path.relative(rootPath, fullPath);

      if (ig.ignores(relPath)) continue;

      if (entry.isDirectory()) {
        if (entry.name === 'node_modules') continue;
        walk(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        if (extSet.size === 0 || extSet.has(ext)) {
          results.push(fullPath);
        }
      }
    }
  }

  walk(rootPath);
  return results.sort();
}

// ---- Main Indexer class ----

export class Indexer {
  private config: IndexerConfig;
  private chunker: ASTChunker;
  private embedder: Embedder;
  private vectorStore: VectorStore;
  private cache: EmbeddingCache;
  private merkleStatePath: string;

  constructor(config: IndexerConfig) {
    this.config = config;
    this.chunker = new ASTChunker(config.maxChunkTokens, config.minChunkTokens);
    this.embedder = new Embedder(config.embeddingModel);
    this.vectorStore = new VectorStore(
      config.qdrantUrl,
      config.collectionName,
      this.embedder.getDimension()
    );
    this.cache = new EmbeddingCache(
      path.resolve(config.rootDir, config.cachePath)
    );
    this.merkleStatePath = path.resolve(
      config.rootDir,
      '.cache',
      'merkle-state.json'
    );
  }

  /**
   * Initialize all components (model loading, collection creation).
   */
  async init(): Promise<void> {
    await loadNative();
    await this.embedder.init();
    await this.vectorStore.ensureCollection();
  }

  /**
   * Run full indexing pipeline (initial or incremental).
   */
  async index(
    onProgress?: (stage: string, detail: string) => void
  ): Promise<IndexStats> {
    const startTime = Date.now();
    const report = (stage: string, detail: string) => {
      onProgress?.(stage, detail);
    };

    // Step 1: Scan files
    report('scan', 'Scanning directory...');
    const filePaths = this.scanFiles();
    report('scan', `Found ${filePaths.length} files`);

    // Step 2: Compute file hashes
    report('hash', 'Computing file hashes...');
    const fileHashes = this.hashFiles(filePaths);
    report('hash', `Hashed ${fileHashes.length} files`);

    // Step 3: Build new Merkle tree
    report('merkle', 'Building Merkle tree...');
    const newMerkleNodes = this.buildMerkle(fileHashes);

    // Step 4: Load old Merkle tree and diff
    const oldMerkleNodes = this.loadMerkleState();
    const diff = this.diffMerkle(oldMerkleNodes, newMerkleNodes);
    const changedFiles = [...diff.added, ...diff.modified];
    const removedFiles = diff.removed;
    report(
      'merkle',
      `Changes: ${diff.added.length} added, ${diff.modified.length} modified, ${diff.removed.length} removed`
    );

    // If no changes and we have an old tree, nothing to do
    if (
      changedFiles.length === 0 &&
      removedFiles.length === 0 &&
      oldMerkleNodes.length > 0
    ) {
      report('done', 'No changes detected, index is up to date');
      return {
        totalFiles: filePaths.length,
        totalChunks: 0,
        newChunks: 0,
        cachedChunks: 0,
        indexTimeMs: Date.now() - startTime,
      };
    }

    // Step 5: Remove vectors for deleted/modified files
    if (removedFiles.length > 0 || diff.modified.length > 0) {
      report('cleanup', 'Removing outdated vectors...');
      await this.vectorStore.deleteByFilePaths([
        ...removedFiles,
        ...diff.modified,
      ]);
    }

    // For initial indexing, process ALL files
    const filesToProcess =
      oldMerkleNodes.length === 0
        ? filePaths
        : changedFiles.map((f) => path.resolve(this.config.rootDir, f));

    // Step 6: Chunk changed files
    report('chunk', `Chunking ${filesToProcess.length} files...`);
    const allChunks: CodeChunk[] = [];
    for (const fp of filesToProcess) {
      try {
        const content = fs.readFileSync(fp, 'utf-8');
        const relPath = path.relative(this.config.rootDir, fp);
        const chunks = this.chunker.chunkFile(relPath, content);
        allChunks.push(...chunks);
      } catch (err) {
        report('chunk', `⚠ Failed to chunk ${fp}: ${err}`);
      }
    }
    report('chunk', `Generated ${allChunks.length} chunks`);

    // Step 7: Separate cached vs uncached chunks
    const uncachedChunks: CodeChunk[] = [];
    const cachedResults: Array<{ chunk: CodeChunk; vector: number[] }> = [];

    for (const chunk of allChunks) {
      const cached = this.cache.get(chunk.contentHash);
      if (cached) {
        cachedResults.push({ chunk, vector: cached });
      } else {
        uncachedChunks.push(chunk);
      }
    }
    report(
      'cache',
      `Cache: ${cachedResults.length} hit, ${uncachedChunks.length} miss`
    );

    // Step 8: Generate embeddings for uncached chunks
    let newEmbeddings: number[][] = [];
    if (uncachedChunks.length > 0) {
      report(
        'embed',
        `Generating embeddings for ${uncachedChunks.length} chunks...`
      );
      const texts = uncachedChunks.map((c) => c.content);
      newEmbeddings = await this.embedder.embedBatch(texts);

      // Save to cache
      for (let i = 0; i < uncachedChunks.length; i++) {
        this.cache.set(uncachedChunks[i].contentHash, newEmbeddings[i]);
      }
    }

    // Step 9: Upsert all vectors into Qdrant
    report('store', 'Upserting vectors into Qdrant...');
    const points = [
      ...cachedResults.map(({ chunk, vector }) => ({
        id: chunk.chunkId,
        vector,
        payload: {
          filePath: chunk.filePath,
          startLine: chunk.startLine,
          endLine: chunk.endLine,
          contentHash: chunk.contentHash,
          nodeType: chunk.nodeType,
          symbolName: chunk.symbolName,
        },
      })),
      ...uncachedChunks.map((chunk, i) => ({
        id: chunk.chunkId,
        vector: newEmbeddings[i],
        payload: {
          filePath: chunk.filePath,
          startLine: chunk.startLine,
          endLine: chunk.endLine,
          contentHash: chunk.contentHash,
          nodeType: chunk.nodeType,
          symbolName: chunk.symbolName,
        },
      })),
    ];

    await this.vectorStore.upsert(points);
    report('store', `Upserted ${points.length} vectors`);

    // Step 10: Save Merkle state and cache
    this.saveMerkleState(newMerkleNodes);
    this.cache.save();

    const stats: IndexStats = {
      totalFiles: filePaths.length,
      totalChunks: allChunks.length,
      newChunks: uncachedChunks.length,
      cachedChunks: cachedResults.length,
      indexTimeMs: Date.now() - startTime,
    };

    report(
      'done',
      `Indexing complete in ${stats.indexTimeMs}ms — ${stats.totalChunks} chunks (${stats.newChunks} new, ${stats.cachedChunks} cached)`
    );

    return stats;
  }

  /**
   * Search the indexed codebase.
   */
  async search(query: string, topK?: number) {
    const queryVector = await this.embedder.embed(query);
    return this.vectorStore.search(queryVector, topK ?? this.config.topK);
  }

  /**
   * Get index statistics.
   */
  async getStats() {
    const qdrantInfo = await this.vectorStore.getInfo();
    const cacheStats = this.cache.stats();
    return {
      qdrant: qdrantInfo,
      cache: cacheStats,
    };
  }

  /**
   * Clean up: delete collection and cache.
   */
  async reset(): Promise<void> {
    await this.vectorStore.deleteCollection();
    this.cache.clear();
    this.cache.save();
    if (fs.existsSync(this.merkleStatePath)) {
      fs.unlinkSync(this.merkleStatePath);
    }
  }

  // ---- Private helpers ----

  private scanFiles(): string[] {
    if (native?.scanDirectory) {
      return native.scanDirectory(
        this.config.rootDir,
        this.config.extensions
      );
    }
    return tsScanDirectory(this.config.rootDir, this.config.extensions);
  }

  private hashFiles(filePaths: string[]): FileHashEntry[] {
    const hashes = native?.sha256HashFiles
      ? native.sha256HashFiles(filePaths)
      : tsSha256HashFiles(filePaths);

    return hashes.map((h) => ({
      path: path.relative(this.config.rootDir, h.path),
      hash: h.hash,
    }));
  }

  private buildMerkle(fileHashes: FileHashEntry[]): MerkleNode[] {
    if (native?.buildMerkleTree) {
      return native.buildMerkleTree(fileHashes);
    }
    return tsBuildMerkleTree(fileHashes);
  }

  private diffMerkle(oldNodes: MerkleNode[], newNodes: MerkleNode[]) {
    if (native?.diffMerkleTrees) {
      return native.diffMerkleTrees(oldNodes, newNodes);
    }
    return tsDiffMerkleTrees(oldNodes, newNodes);
  }

  private loadMerkleState(): MerkleNode[] {
    try {
      if (fs.existsSync(this.merkleStatePath)) {
        const raw = fs.readFileSync(this.merkleStatePath, 'utf-8');
        return JSON.parse(raw) as MerkleNode[];
      }
    } catch {
      // Corrupted state, start fresh
    }
    return [];
  }

  private saveMerkleState(nodes: MerkleNode[]): void {
    const dir = path.dirname(this.merkleStatePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(this.merkleStatePath, JSON.stringify(nodes), 'utf-8');
  }
}
