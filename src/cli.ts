#!/usr/bin/env node

/**
 * CLI entry point for the codebase indexer.
 *
 * Commands:
 *   index [dir]        Index a codebase directory
 *   search [query]     Search the indexed codebase
 *   stats              Show index statistics
 *   reset              Delete index and cache
 */

import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import path from 'node:path';
import fs from 'node:fs';
import { Indexer } from './indexer.js';
import { Retriever } from './search/retriever.js';
import { Embedder } from './embedding/embedder.js';
import { VectorStore } from './storage/vector-store.js';
import type { IndexerConfig } from './types.js';
import { DEFAULT_CONFIG } from './types.js';
import readline from 'node:readline';

function resolveConfig(rootDir: string, opts: Partial<IndexerConfig>): IndexerConfig {
  return {
    rootDir: path.resolve(rootDir),
    extensions: opts.extensions ?? (DEFAULT_CONFIG.extensions as string[]),
    qdrantUrl: opts.qdrantUrl ?? (DEFAULT_CONFIG.qdrantUrl as string),
    collectionName: opts.collectionName ?? (DEFAULT_CONFIG.collectionName as string),
    embeddingModel: opts.embeddingModel ?? (DEFAULT_CONFIG.embeddingModel as string),
    maxChunkTokens: opts.maxChunkTokens ?? (DEFAULT_CONFIG.maxChunkTokens as number),
    minChunkTokens: opts.minChunkTokens ?? (DEFAULT_CONFIG.minChunkTokens as number),
    cachePath: opts.cachePath ?? (DEFAULT_CONFIG.cachePath as string),
    topK: opts.topK ?? (DEFAULT_CONFIG.topK as number),
  };
}

const program = new Command();

program
  .name('codebase-indexer')
  .description('Local codebase indexing and semantic search engine')
  .version('0.1.0');

// ---- INDEX command ----
program
  .command('index')
  .description('Index a codebase directory')
  .argument('[dir]', 'Directory to index', '.')
  .option('--extensions <exts>', 'Comma-separated file extensions', '.ts,.tsx')
  .option('--qdrant-url <url>', 'Qdrant server URL', DEFAULT_CONFIG.qdrantUrl)
  .option('--collection <name>', 'Qdrant collection name', DEFAULT_CONFIG.collectionName)
  .option('--model <name>', 'Embedding model name', DEFAULT_CONFIG.embeddingModel)
  .option('--reset', 'Delete existing index before re-indexing', false)
  .action(async (dir: string, opts: Record<string, string | boolean>) => {
    const config = resolveConfig(dir, {
      extensions: typeof opts.extensions === 'string'
        ? opts.extensions.split(',').map((e) => (e.startsWith('.') ? e : `.${e}`))
        : undefined,
      qdrantUrl: opts.qdrantUrl as string | undefined,
      collectionName: opts.collection as string | undefined,
      embeddingModel: opts.model as string | undefined,
    });

    console.log(chalk.bold.cyan('\n🔍 Codebase Indexer\n'));
    console.log(chalk.gray(`  Root:       ${config.rootDir}`));
    console.log(chalk.gray(`  Extensions: ${config.extensions.join(', ')}`));
    console.log(chalk.gray(`  Qdrant:     ${config.qdrantUrl}`));
    console.log(chalk.gray(`  Collection: ${config.collectionName}`));
    console.log(chalk.gray(`  Model:      ${config.embeddingModel}`));
    console.log('');

    const spinner = ora('Initializing...').start();

    try {
      const indexer = new Indexer(config);

      if (opts.reset) {
        spinner.text = 'Resetting index...';
        await indexer.reset();
      }

      spinner.text = 'Loading embedding model (first run downloads ~440MB)...';
      await indexer.init();

      const stats = await indexer.index((stage, detail) => {
        spinner.text = `[${stage}] ${detail}`;
      });

      spinner.succeed(chalk.green('Indexing complete!'));
      console.log('');
      console.log(chalk.bold('  📊 Statistics:'));
      console.log(chalk.gray(`     Files:          ${stats.totalFiles}`));
      console.log(chalk.gray(`     Total chunks:   ${stats.totalChunks}`));
      console.log(chalk.gray(`     New embeddings: ${stats.newChunks}`));
      console.log(chalk.gray(`     Cache hits:     ${stats.cachedChunks}`));
      console.log(chalk.gray(`     Time:           ${stats.indexTimeMs}ms`));
      console.log('');
    } catch (err) {
      spinner.fail(chalk.red(`Indexing failed: ${err}`));
      process.exit(1);
    }
  });

// ---- SEARCH command ----
program
  .command('search')
  .description('Search the indexed codebase')
  .argument('[query]', 'Search query')
  .option('-k, --top-k <n>', 'Number of results', '5')
  .option('--dir <dir>', 'Root directory of the indexed codebase', '.')
  .option('--qdrant-url <url>', 'Qdrant server URL', DEFAULT_CONFIG.qdrantUrl)
  .option('--collection <name>', 'Qdrant collection name', DEFAULT_CONFIG.collectionName)
  .option('--model <name>', 'Embedding model name', DEFAULT_CONFIG.embeddingModel)
  .option('-i, --interactive', 'Interactive search mode', false)
  .action(async (query: string | undefined, opts: Record<string, string | boolean>) => {
    const config = resolveConfig(opts.dir as string ?? '.', {
      qdrantUrl: opts.qdrantUrl as string | undefined,
      collectionName: opts.collection as string | undefined,
      embeddingModel: opts.model as string | undefined,
      topK: Number(opts.topK) || 5,
    });

    const embedder = new Embedder(config.embeddingModel);
    const vectorStore = new VectorStore(
      config.qdrantUrl,
      config.collectionName,
      embedder.getDimension()
    );
    const retriever = new Retriever(config.rootDir, embedder, vectorStore);

    const spinner = ora('Loading embedding model...').start();
    await embedder.init();
    spinner.succeed('Model loaded');

    if (opts.interactive || !query) {
      // Interactive mode
      console.log(chalk.bold.cyan('\n🔍 Interactive Search Mode'));
      console.log(chalk.gray('  Type your query and press Enter. Type "exit" to quit.\n'));

      const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
      });

      const askQuestion = () => {
        rl.question(chalk.cyan('🔎 Query: '), async (input) => {
          const q = input.trim();
          if (q === 'exit' || q === 'quit' || q === '') {
            rl.close();
            return;
          }

          const searchSpinner = ora('Searching...').start();
          try {
            const results = await retriever.search(q, config.topK);
            searchSpinner.stop();
            printResults(results);
          } catch (err) {
            searchSpinner.fail(`Search failed: ${err}`);
          }

          askQuestion();
        });
      };

      askQuestion();
    } else {
      // Single query mode
      const searchSpinner = ora('Searching...').start();
      try {
        const results = await retriever.search(query, config.topK);
        searchSpinner.stop();
        printResults(results);
      } catch (err) {
        searchSpinner.fail(`Search failed: ${err}`);
        process.exit(1);
      }
    }
  });

// ---- STATS command ----
program
  .command('stats')
  .description('Show index statistics')
  .option('--dir <dir>', 'Root directory', '.')
  .option('--qdrant-url <url>', 'Qdrant server URL', DEFAULT_CONFIG.qdrantUrl)
  .option('--collection <name>', 'Qdrant collection name', DEFAULT_CONFIG.collectionName)
  .action(async (opts: Record<string, string>) => {
    const config = resolveConfig(opts.dir ?? '.', {
      qdrantUrl: opts.qdrantUrl,
      collectionName: opts.collection,
    });

    try {
      const indexer = new Indexer(config);
      const stats = await indexer.getStats();

      console.log(chalk.bold.cyan('\n📊 Index Statistics\n'));
      console.log(chalk.gray(`  Qdrant vectors: ${stats.qdrant.pointCount}`));
      console.log(chalk.gray(`  Cache entries:  ${stats.cache.entries}`));
      console.log(
        chalk.gray(
          `  Cache size:     ${(stats.cache.sizeBytes / 1024).toFixed(1)} KB`
        )
      );
      console.log('');
    } catch (err) {
      console.error(chalk.red(`Failed to get stats: ${err}`));
      process.exit(1);
    }
  });

// ---- RESET command ----
program
  .command('reset')
  .description('Delete index and cache')
  .option('--dir <dir>', 'Root directory', '.')
  .option('--qdrant-url <url>', 'Qdrant server URL', DEFAULT_CONFIG.qdrantUrl)
  .option('--collection <name>', 'Qdrant collection name', DEFAULT_CONFIG.collectionName)
  .action(async (opts: Record<string, string>) => {
    const config = resolveConfig(opts.dir ?? '.', {
      qdrantUrl: opts.qdrantUrl,
      collectionName: opts.collection,
    });

    try {
      const indexer = new Indexer(config);
      await indexer.reset();
      console.log(chalk.green('\n✅ Index and cache cleared.\n'));
    } catch (err) {
      console.error(chalk.red(`Failed to reset: ${err}`));
      process.exit(1);
    }
  });

// ---- Helpers ----

function printResults(results: Array<{
  filePath: string;
  startLine: number;
  endLine: number;
  score: number;
  content: string;
  nodeType: string;
  symbolName?: string;
}>) {
  if (results.length === 0) {
    console.log(chalk.yellow('\n  No results found.\n'));
    return;
  }

  console.log(chalk.bold(`\n  Found ${results.length} results:\n`));

  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    const header = chalk.bold.white(
      `  #${i + 1} `
    ) + chalk.cyan(r.filePath) + chalk.gray(`:${r.startLine}-${r.endLine}`) +
      chalk.green(` (score: ${r.score.toFixed(3)})`) +
      (r.symbolName ? chalk.yellow(` [${r.symbolName}]`) : '');

    console.log(header);
    console.log(chalk.gray('  ' + '─'.repeat(70)));

    // Print code with line numbers
    const lines = r.content.split('\n');
    for (let j = 0; j < lines.length; j++) {
      const lineNum = (r.startLine + j).toString().padStart(4, ' ');
      console.log(chalk.gray(`  ${lineNum} │ `) + lines[j]);
    }
    console.log('');
  }
}

program.parse();
