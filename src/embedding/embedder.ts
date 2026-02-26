/**
 * Embedding module supporting multiple backends:
 *
 * 1. **OpenAI API** (primary when API key is provided)
 *    - Model: text-embedding-3-small (1536-dimensional)
 *    - Fast, high quality, requires network + API key
 *
 * 2. **Local HuggingFace / ONNX** (fallback)
 *    - Model: Xenova/bge-m3 (1024-dimensional, q4 quantized)
 *    - Runs entirely offline, no API key needed
 *    - Higher memory usage (~600MB for q4)
 *
 * Strategy:
 *   - If `openaiApiKey` is set, probe OpenAI at init().
 *   - If probe succeeds → use OpenAI for all embeddings.
 *   - If probe fails → log warning, fall back to local model.
 *   - If no API key → use local model directly.
 */

import { pipeline, type FeatureExtractionPipeline } from '@huggingface/transformers';

/** Supported quantization types for ONNX models */
export type ModelDtype = 'fp32' | 'fp16' | 'q8' | 'q4';

/** Options for the Embedder constructor */
export interface EmbedderOptions {
  /** OpenAI API key. If provided, OpenAI is tried first. */
  openaiApiKey?: string;
  /** OpenAI model name (default: text-embedding-3-small) */
  openaiModel?: string;
  /** Quantization dtype for local ONNX model (default: q4) */
  dtype?: ModelDtype;
}

/**
 * Known embedding dimensions for common models.
 * Used by getDimension() before init() for VectorStore creation.
 */
const KNOWN_DIMENSIONS: Record<string, number> = {
  'Xenova/bge-base-en-v1.5': 768,
  'BAAI/bge-base-en-v1.5': 768,
  'Xenova/bge-small-en-v1.5': 384,
  'BAAI/bge-small-en-v1.5': 384,
  'Xenova/bge-large-en-v1.5': 1024,
  'BAAI/bge-large-en-v1.5': 1024,
  'Xenova/bge-m3': 1024,
  'BAAI/bge-m3': 1024,
  'text-embedding-3-small': 1536,
  'text-embedding-3-large': 3072,
  'text-embedding-ada-002': 1536,
};

export class Embedder {
  // ---- Backend state ----
  private mode: 'openai' | 'local' | 'uninitialized' = 'uninitialized';

  // ---- OpenAI ----
  private openaiApiKey?: string;
  private openaiModel: string;

  // ---- Local HuggingFace ----
  private localModel: FeatureExtractionPipeline | null = null;
  private localModelName: string;
  private dtype: ModelDtype;

  // ---- Shared ----
  private initPromise: Promise<void> | null = null;
  private detectedDimension: number | null = null;

  constructor(localModelName = 'Xenova/bge-m3', options?: EmbedderOptions) {
    this.localModelName = localModelName;
    this.dtype = options?.dtype ?? 'q4';
    this.openaiApiKey = options?.openaiApiKey;
    this.openaiModel = options?.openaiModel ?? 'text-embedding-3-small';
  }

  /**
   * Initialize the embedder.
   * - If openaiApiKey is provided, probes OpenAI first.
   * - Falls back to local ONNX model on failure.
   */
  async init(): Promise<void> {
    if (this.mode !== 'uninitialized') return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = this._doInit();
    return this.initPromise;
  }

  private async _doInit(): Promise<void> {
    // Try OpenAI first if API key is available
    if (this.openaiApiKey) {
      try {
        console.log(`  ⏳ Probing OpenAI API (${this.openaiModel})...`);
        const probeResult = await this._openaiEmbed(['hello']);
        this.mode = 'openai';
        this.detectedDimension = probeResult[0].length;
        console.log(
          `  ✓ Using OpenAI embeddings (${this.openaiModel}, ${this.detectedDimension}d)`
        );
        return;
      } catch (err) {
        console.warn(
          `  ⚠ OpenAI embedding failed: ${err instanceof Error ? err.message : err}`
        );
        console.warn(`  ⚠ Falling back to local model (${this.localModelName})...`);
      }
    }

    // Fall back to local model
    console.log(`  ⏳ Loading local model (${this.localModelName}, ${this.dtype})...`);
    this.localModel = await pipeline('feature-extraction', this.localModelName, {
      dtype: this.dtype,
    });

    // Auto-detect dimension via probe
    const probe = await this.localModel('hello', {
      pooling: 'cls',
      normalize: true,
    });
    this.detectedDimension = (probe.data as Float32Array).length;
    this.mode = 'local';
    console.log(
      `  ✓ Using local embeddings (${this.localModelName}, ${this.detectedDimension}d)`
    );
  }

  /**
   * Generate embedding for a single text.
   */
  async embed(text: string): Promise<number[]> {
    await this.init();
    if (this.mode === 'openai') {
      const [vec] = await this._openaiEmbed([text]);
      return vec;
    }
    return this._localEmbed(text);
  }

  /**
   * Generate embeddings for multiple texts in batch.
   */
  async embedBatch(texts: string[], batchSize = 32): Promise<number[][]> {
    await this.init();
    if (this.mode === 'openai') {
      return this._openaiBatchEmbed(texts, batchSize);
    }
    return this._localBatchEmbed(texts, batchSize);
  }

  /**
   * Get the dimensionality of the embeddings.
   *
   * Before init(): returns best-guess from lookup table.
   * After init(): returns actual detected dimension.
   */
  getDimension(): number {
    if (this.detectedDimension !== null) {
      return this.detectedDimension;
    }

    // Before init, try to guess from config
    if (this.openaiApiKey) {
      const dim = KNOWN_DIMENSIONS[this.openaiModel];
      if (dim) return dim;
    }

    const known = KNOWN_DIMENSIONS[this.localModelName];
    if (known !== undefined) return known;

    throw new Error(
      `Unknown embedding dimension for model "${this.localModelName}". ` +
        `Either add it to KNOWN_DIMENSIONS or call init() first.`
    );
  }

  /**
   * Returns which backend is active after init().
   */
  getMode(): string {
    return this.mode;
  }

  // ─── OpenAI internals ─────────────────────────────────────────────

  /**
   * Truncate text to fit within the OpenAI embedding model's context window.
   * text-embedding-3-small has a max of 8191 tokens.
   * We use a conservative ~3 chars/token estimate → ~24K char limit.
   */
  private _truncateForOpenAI(text: string): string {
    const MAX_CHARS = 24_000; // ~8000 tokens at 3 chars/token
    if (text.length <= MAX_CHARS) return text;
    return text.slice(0, MAX_CHARS);
  }

  private async _openaiEmbed(texts: string[]): Promise<number[][]> {
    // Ensure no single input exceeds the model's context window
    const safeTexts = texts.map((t) => this._truncateForOpenAI(t));

    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.openaiApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.openaiModel,
        input: safeTexts,
      }),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`OpenAI API error ${response.status}: ${body}`);
    }

    const json = (await response.json()) as {
      data: Array<{ embedding: number[]; index: number }>;
    };

    // OpenAI may return results out of order; sort by index
    const sorted = json.data.sort((a, b) => a.index - b.index);
    return sorted.map((d) => d.embedding);
  }

  /**
   * Conservative token estimate for OpenAI batching.
   * Code has many short symbols (brackets, operators, dots) that each consume
   * a full token but only 1-2 characters, so we use ~3 chars/token instead of 4.
   */
  private _estimateTokens(text: string): number {
    return Math.ceil(text.length / 3);
  }

  private async _openaiBatchEmbed(
    texts: string[],
    _batchSize: number
  ): Promise<number[][]> {
    const all: number[][] = [];

    // OpenAI counts total tokens across ALL inputs per request.
    // text-embedding-3-small context limit = 8191 tokens.
    // We use a conservative budget to account for tokenizer differences.
    const MAX_TOKENS_PER_REQUEST = 6000;
    const MAX_INPUTS_PER_REQUEST = 2048; // OpenAI hard limit on input count

    let batchStart = 0;
    while (batchStart < texts.length) {
      const batch: string[] = [];
      let batchTokens = 0;

      for (let i = batchStart; i < texts.length; i++) {
        const tokensForThis = this._estimateTokens(texts[i]);

        // If a single text exceeds the budget, send it alone (OpenAI will
        // truncate or error, but at least we don't block other chunks)
        if (batch.length === 0) {
          batch.push(texts[i]);
          batchTokens += tokensForThis;
          continue;
        }

        if (
          batchTokens + tokensForThis > MAX_TOKENS_PER_REQUEST ||
          batch.length >= MAX_INPUTS_PER_REQUEST
        ) {
          break;
        }

        batch.push(texts[i]);
        batchTokens += tokensForThis;
      }

      const embeddings = await this._openaiEmbed(batch);
      all.push(...embeddings);
      batchStart += batch.length;
    }

    return all;
  }

  // ─── Local model internals ────────────────────────────────────────

  private async _localEmbed(text: string): Promise<number[]> {
    if (!this.localModel) throw new Error('Local model not initialized');
    const result = await this.localModel(text, {
      pooling: 'cls',
      normalize: true,
    });
    return Array.from(result.data as Float32Array);
  }

  private async _localBatchEmbed(
    texts: string[],
    batchSize: number
  ): Promise<number[][]> {
    if (!this.localModel) throw new Error('Local model not initialized');
    const all: number[][] = [];

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const results = await this.localModel(batch, {
        pooling: 'cls',
        normalize: true,
      });

      const data = results.data as Float32Array;
      const dim = data.length / batch.length;

      for (let j = 0; j < batch.length; j++) {
        const start = j * dim;
        const end = start + dim;
        all.push(Array.from(data.slice(start, end)));
      }
    }

    return all;
  }
}
