/**
 * Search Retriever: Performs semantic search and reads actual code from local disk.
 *
 * Flow:
 * 1. Query → embedding
 * 2. Vector search in Qdrant → returns metadata (file path, line range, score)
 * 3. Read actual code from local file system using metadata
 * 4. Return enriched results with code content
 */

import fs from 'node:fs';
import path from 'node:path';
import type { SearchResult } from '../types.js';
import { Embedder } from '../embedding/embedder.js';
import { VectorStore, type VectorSearchResult } from '../storage/vector-store.js';

export class Retriever {
  private embedder: Embedder;
  private vectorStore: VectorStore;
  private rootDir: string;

  constructor(
    rootDir: string,
    embedder: Embedder,
    vectorStore: VectorStore
  ) {
    this.rootDir = rootDir;
    this.embedder = embedder;
    this.vectorStore = vectorStore;
  }

  /**
   * Search the codebase with a natural language query.
   * Returns code snippets with full source content read from local disk.
   */
  async search(query: string, topK = 10): Promise<SearchResult[]> {
    // Step 1: Embed the query
    const queryVector = await this.embedder.embed(query);

    // Step 2: Search Qdrant
    const vectorResults = await this.vectorStore.search(queryVector, topK);

    // Step 3: Enrich with actual code content from disk
    const results = await this.enrichResults(vectorResults);

    return results;
  }

  /**
   * Read actual code content from local files and attach to search results.
   */
  private async enrichResults(
    vectorResults: VectorSearchResult[]
  ): Promise<SearchResult[]> {
    const results: SearchResult[] = [];

    for (const vr of vectorResults) {
      const { filePath, startLine, endLine, nodeType, symbolName } = vr.payload;
      const absPath = path.resolve(this.rootDir, filePath);

      let content = '';
      try {
        if (fs.existsSync(absPath)) {
          const fileContent = fs.readFileSync(absPath, 'utf-8');
          const lines = fileContent.split('\n');

          // Extract the specific line range (1-based to 0-based conversion)
          const start = Math.max(0, startLine - 1);
          const end = Math.min(lines.length, endLine);
          content = lines.slice(start, end).join('\n');
        } else {
          content = `[File not found: ${filePath}]`;
        }
      } catch (err) {
        content = `[Error reading file: ${err}]`;
      }

      results.push({
        filePath,
        startLine,
        endLine,
        score: vr.score,
        content,
        nodeType,
        symbolName: symbolName ?? undefined,
      });
    }

    return results;
  }

  /**
   * Format search results as a context string for LLM prompts.
   */
  formatForLLM(results: SearchResult[]): string {
    if (results.length === 0) {
      return 'No relevant code found.';
    }

    const sections = results.map((r, i) => {
      const header = `--- #${i + 1} ${r.filePath}:${r.startLine}-${r.endLine} (score: ${r.score.toFixed(3)})${r.symbolName ? ` [${r.symbolName}]` : ''} ---`;
      return `${header}\n${r.content}`;
    });

    return sections.join('\n\n');
  }
}
