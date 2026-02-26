/**
 * Vector Store: Qdrant client wrapper for storing and searching code embeddings.
 *
 * Point IDs: Qdrant requires UUIDs or unsigned integers as point IDs.
 * We convert SHA-256 content hashes to UUID v5 format deterministically,
 * preserving content-addressability. The full SHA-256 is stored in the payload.
 */

import { QdrantClient } from '@qdrant/js-client-rest';
import crypto from 'node:crypto';

/**
 * Convert a SHA-256 hex string to a UUID v5-like format.
 * Takes the first 32 hex chars (128 bits) and formats as UUID,
 * setting version and variant bits per RFC 4122.
 */
function sha256ToUuid(sha256Hex: string): string {
  // Take first 32 hex chars (128 bits)
  const hex = sha256Hex.substring(0, 32);
  // Format as UUID: 8-4-4-4-12
  let uuid =
    hex.substring(0, 8) +
    '-' +
    hex.substring(8, 12) +
    '-' +
    hex.substring(12, 16) +
    '-' +
    hex.substring(16, 20) +
    '-' +
    hex.substring(20, 32);

  // Set version nibble (position 12) to 5 (UUID v5)
  const chars = uuid.split('');
  chars[14] = '5'; // version nibble
  // Set variant bits (position 16) to 8, 9, a, or b
  const variantChar = parseInt(chars[19], 16);
  chars[19] = ((variantChar & 0x3) | 0x8).toString(16);
  return chars.join('');
}

export interface VectorPayload {
  filePath: string;
  startLine: number;
  endLine: number;
  contentHash: string;
  nodeType: string;
  symbolName?: string;
}

export interface VectorSearchResult {
  id: string;
  score: number;
  payload: VectorPayload;
}

export class VectorStore {
  private client: QdrantClient;
  private collectionName: string;
  private dimension: number;

  constructor(url: string, collectionName: string, dimension: number) {
    this.client = new QdrantClient({ url });
    this.collectionName = collectionName;
    this.dimension = dimension;
  }

  /**
   * Ensure the Qdrant collection exists with the correct configuration.
   */
  async ensureCollection(): Promise<void> {
    try {
      await this.client.getCollection(this.collectionName);
    } catch {
      // Collection doesn't exist, create it
      await this.client.createCollection(this.collectionName, {
        vectors: {
          size: this.dimension,
          distance: 'Cosine',
        },
      });

      // Create payload index on filePath for efficient filtering
      await this.client.createPayloadIndex(this.collectionName, {
        field_name: 'filePath',
        field_schema: 'keyword',
      });
    }
  }

  /**
   * Upsert vectors with payloads.
   * Point IDs are SHA-256 content hashes converted to UUID format.
   */
  async upsert(
    points: Array<{
      id: string; // contentHash (SHA-256 hex)
      vector: number[];
      payload: VectorPayload;
    }>
  ): Promise<void> {
    if (points.length === 0) return;

    // Qdrant accepts batches; chunk into groups of 100
    const batchSize = 100;
    for (let i = 0; i < points.length; i += batchSize) {
      const batch = points.slice(i, i + batchSize);
      await this.client.upsert(this.collectionName, {
        wait: true,
        points: batch.map((p) => ({
          id: sha256ToUuid(p.id),
          vector: p.vector,
          payload: p.payload as unknown as Record<string, unknown>,
        })),
      });
    }
  }

  /**
   * Delete vectors by file paths (used when files are removed or modified).
   */
  async deleteByFilePaths(filePaths: string[]): Promise<void> {
    if (filePaths.length === 0) return;

    await this.client.delete(this.collectionName, {
      wait: true,
      filter: {
        must: [
          {
            key: 'filePath',
            match: {
              any: filePaths,
            },
          },
        ],
      },
    });
  }

  /**
   * Delete vectors by IDs (SHA-256 content hashes, converted to UUID).
   */
  async deleteByIds(ids: string[]): Promise<void> {
    if (ids.length === 0) return;

    await this.client.delete(this.collectionName, {
      wait: true,
      points: ids.map(sha256ToUuid),
    });
  }

  /**
   * Search for similar vectors.
   */
  async search(
    queryVector: number[],
    topK = 10
  ): Promise<VectorSearchResult[]> {
    const results = await this.client.search(this.collectionName, {
      vector: queryVector,
      limit: topK,
      with_payload: true,
    });

    return results.map((r) => ({
      id: String(r.id),
      score: r.score,
      payload: r.payload as unknown as VectorPayload,
    }));
  }

  /**
   * Get collection info (point count, etc.).
   */
  async getInfo(): Promise<{ pointCount: number; status: string }> {
    try {
      const info = await this.client.getCollection(this.collectionName);
      return {
        pointCount: info.points_count ?? 0,
        status: info.status,
      };
    } catch {
      return { pointCount: 0, status: 'not_found' };
    }
  }

  /**
   * Delete the entire collection.
   */
  async deleteCollection(): Promise<void> {
    try {
      await this.client.deleteCollection(this.collectionName);
    } catch {
      // Collection may not exist
    }
  }

  /**
   * Scroll through all points (for debugging/export).
   */
  async scrollAll(limit = 100) {
    return this.client.scroll(this.collectionName, {
      limit,
      with_payload: true,
      with_vector: false,
    });
  }
}
