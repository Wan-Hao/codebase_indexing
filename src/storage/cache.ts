/**
 * Content-addressable embedding cache.
 *
 * Maps SHA-256 content hashes to embedding vectors.
 * Persisted to disk as a JSON file so embeddings survive across runs.
 * Avoids redundant embedding model inference for unchanged code chunks.
 */

import fs from 'node:fs';
import path from 'node:path';

interface CacheEntry {
  vector: number[];
  timestamp: number;
}

export class EmbeddingCache {
  private store = new Map<string, CacheEntry>();
  private filePath: string;
  private dirty = false;

  constructor(cachePath: string) {
    this.filePath = cachePath;
    this.load();
  }

  /**
   * Get a cached embedding by content hash.
   * Returns the vector if found, null otherwise.
   */
  get(contentHash: string): number[] | null {
    const entry = this.store.get(contentHash);
    return entry?.vector ?? null;
  }

  /**
   * Store an embedding for a content hash.
   */
  set(contentHash: string, vector: number[]): void {
    this.store.set(contentHash, {
      vector,
      timestamp: Date.now(),
    });
    this.dirty = true;
  }

  /**
   * Check if a content hash is cached.
   */
  has(contentHash: string): boolean {
    return this.store.has(contentHash);
  }

  /**
   * Persist cache to disk.
   */
  save(): void {
    if (!this.dirty && fs.existsSync(this.filePath)) return;

    const dir = path.dirname(this.filePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    const data: Record<string, CacheEntry> = {};
    for (const [key, value] of this.store) {
      data[key] = value;
    }

    fs.writeFileSync(this.filePath, JSON.stringify(data), 'utf-8');
    this.dirty = false;
  }

  /**
   * Load cache from disk.
   */
  private load(): void {
    try {
      if (fs.existsSync(this.filePath)) {
        const raw = fs.readFileSync(this.filePath, 'utf-8');
        const data = JSON.parse(raw) as Record<string, CacheEntry>;
        for (const [key, value] of Object.entries(data)) {
          this.store.set(key, value);
        }
      }
    } catch {
      // Corrupted cache, start fresh
      this.store.clear();
    }
  }

  /**
   * Clear the entire cache.
   */
  clear(): void {
    this.store.clear();
    this.dirty = true;
  }

  /**
   * Get cache statistics.
   */
  stats(): { entries: number; sizeBytes: number } {
    let sizeBytes = 0;
    try {
      if (fs.existsSync(this.filePath)) {
        sizeBytes = fs.statSync(this.filePath).size;
      }
    } catch {
      // ignore
    }
    return {
      entries: this.store.size,
      sizeBytes,
    };
  }

  /**
   * Remove entries older than maxAgeMs.
   */
  prune(maxAgeMs: number): number {
    const now = Date.now();
    let removed = 0;
    for (const [key, entry] of this.store) {
      if (now - entry.timestamp > maxAgeMs) {
        this.store.delete(key);
        removed++;
      }
    }
    if (removed > 0) this.dirty = true;
    return removed;
  }
}
