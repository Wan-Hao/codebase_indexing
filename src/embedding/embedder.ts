/**
 * Embedding module using @huggingface/transformers.
 *
 * Runs BAAI/bge-base-en-v1.5 (ONNX) locally in Node.js.
 * Produces 768-dimensional normalized embeddings.
 */

import { pipeline, type FeatureExtractionPipeline } from '@huggingface/transformers';

export class Embedder {
  private model: FeatureExtractionPipeline | null = null;
  private modelName: string;
  private initPromise: Promise<void> | null = null;

  constructor(modelName = 'Xenova/bge-base-en-v1.5') {
    this.modelName = modelName;
  }

  /**
   * Initialize the model (downloads on first run, cached after that).
   * Call this once before generating embeddings.
   */
  async init(): Promise<void> {
    if (this.model) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      this.model = await pipeline('feature-extraction', this.modelName, {
        dtype: 'fp32',
      });
    })();

    return this.initPromise;
  }

  /**
   * Generate embedding for a single text.
   * Returns a normalized 768-dimensional vector.
   */
  async embed(text: string): Promise<number[]> {
    await this.init();
    if (!this.model) throw new Error('Model not initialized');

    const result = await this.model(text, {
      pooling: 'cls',
      normalize: true,
    });

    // result is a Tensor; convert to flat array
    return Array.from(result.data as Float32Array);
  }

  /**
   * Generate embeddings for multiple texts in batch.
   * More efficient than calling embed() in a loop.
   */
  async embedBatch(texts: string[], batchSize = 32): Promise<number[][]> {
    await this.init();
    if (!this.model) throw new Error('Model not initialized');

    const allEmbeddings: number[][] = [];

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const results = await this.model(batch, {
        pooling: 'cls',
        normalize: true,
      });

      // results.data is a flat Float32Array of shape [batch_size, dim]
      const data = results.data as Float32Array;
      const dim = data.length / batch.length;

      for (let j = 0; j < batch.length; j++) {
        const start = j * dim;
        const end = start + dim;
        allEmbeddings.push(Array.from(data.slice(start, end)));
      }
    }

    return allEmbeddings;
  }

  /**
   * Get the dimensionality of the embeddings.
   */
  getDimension(): number {
    // bge-base-en-v1.5 outputs 768-dimensional vectors
    return 768;
  }
}
