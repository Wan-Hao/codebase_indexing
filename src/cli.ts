#!/usr/bin/env node

/**
 * CLI entry point for the codebase indexer.
 *
 * Commands:
 *   index [dir]        Index a codebase directory
 *   search [query]     Search the indexed codebase
 *   stats              Show index statistics
 *   reset              Delete index and cache
 *
 * Supports .env file for configuration (loaded from cwd).
 * See .env.example for available variables.
 */

import 'dotenv/config';
import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import path from 'node:path';
import fs from 'node:fs';
import { highlight } from 'cli-highlight';
import { Indexer } from './indexer.js';
import { Retriever } from './search/retriever.js';
import { Embedder } from './embedding/embedder.js';
import { VectorStore } from './storage/vector-store.js';
import type { IndexerConfig } from './types.js';
import { DEFAULT_CONFIG } from './types.js';
import readline from 'node:readline';

/** Resolve the default target directory from env or fallback to cwd */
const DEFAULT_DIR = process.env.INDEX_DIR || '.';

function resolveConfig(rootDir: string, opts: Partial<IndexerConfig>): IndexerConfig {
  return {
    rootDir: path.resolve(rootDir),
    extensions: opts.extensions ?? (DEFAULT_CONFIG.extensions as string[]),
    qdrantUrl: opts.qdrantUrl ?? process.env.QDRANT_URL ?? (DEFAULT_CONFIG.qdrantUrl as string),
    collectionName: opts.collectionName ?? process.env.QDRANT_COLLECTION ?? (DEFAULT_CONFIG.collectionName as string),
    embeddingModel: opts.embeddingModel ?? process.env.EMBEDDING_MODEL ?? (DEFAULT_CONFIG.embeddingModel as string),
    openaiApiKey: opts.openaiApiKey ?? process.env.OPENAI_API_KEY,
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
  .argument('[dir]', 'Directory to index (or set INDEX_DIR env var)', DEFAULT_DIR)
  .option('--extensions <exts>', 'Comma-separated file extensions', '.ts,.tsx')
  .option('--qdrant-url <url>', 'Qdrant server URL', DEFAULT_CONFIG.qdrantUrl)
  .option('--collection <name>', 'Qdrant collection name', DEFAULT_CONFIG.collectionName)
  .option('--model <name>', 'Embedding model name', DEFAULT_CONFIG.embeddingModel)
  .option('--openai-key <key>', 'OpenAI API key (or set OPENAI_API_KEY env var)')
  .option('--reset', 'Delete existing index before re-indexing', false)
  .action(async (dir: string, opts: Record<string, string | boolean>) => {
    const config = resolveConfig(dir, {
      extensions: typeof opts.extensions === 'string'
        ? opts.extensions.split(',').map((e) => (e.startsWith('.') ? e : `.${e}`))
        : undefined,
      qdrantUrl: opts.qdrantUrl as string | undefined,
      collectionName: opts.collection as string | undefined,
      embeddingModel: opts.model as string | undefined,
      openaiApiKey: opts.openaiKey as string | undefined,
    });

    console.log(chalk.bold.cyan('\n🔍 Codebase Indexer\n'));
    console.log(chalk.gray(`  Root:       ${config.rootDir}`));
    console.log(chalk.gray(`  Extensions: ${config.extensions.join(', ')}`));
    console.log(chalk.gray(`  Qdrant:     ${config.qdrantUrl}`));
    console.log(chalk.gray(`  Collection: ${config.collectionName}`));
    console.log(chalk.gray(`  Model:      ${config.openaiApiKey ? 'OpenAI text-embedding-3-small (→ local fallback)' : config.embeddingModel}`));
    console.log('');

    const spinner = ora('Initializing...').start();

    try {
      const indexer = new Indexer(config);

      if (opts.reset) {
        spinner.text = 'Resetting index...';
        await indexer.reset();
      }

      spinner.text = 'Loading embedding model (first run downloads model weights)...';
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
  .option('--dir <dir>', 'Root directory of the indexed codebase (or set INDEX_DIR env var)', DEFAULT_DIR)
  .option('--qdrant-url <url>', 'Qdrant server URL', DEFAULT_CONFIG.qdrantUrl)
  .option('--collection <name>', 'Qdrant collection name', DEFAULT_CONFIG.collectionName)
  .option('--model <name>', 'Embedding model name', DEFAULT_CONFIG.embeddingModel)
  .option('--openai-key <key>', 'OpenAI API key (or set OPENAI_API_KEY env var)')
  .option('-i, --interactive', 'Interactive search mode', false)
  .action(async (query: string | undefined, opts: Record<string, string | boolean>) => {
    const config = resolveConfig(opts.dir as string ?? '.', {
      qdrantUrl: opts.qdrantUrl as string | undefined,
      collectionName: opts.collection as string | undefined,
      embeddingModel: opts.model as string | undefined,
      openaiApiKey: opts.openaiKey as string | undefined,
      topK: Number(opts.topK) || 5,
    });

    const embedder = new Embedder(config.embeddingModel, {
      openaiApiKey: config.openaiApiKey,
    });

    const spinner = ora('Loading embedding model...').start();
    await embedder.init();
    spinner.succeed(`Model loaded (${embedder.getMode()})`);

    const vectorStore = new VectorStore(
      config.qdrantUrl,
      config.collectionName,
      embedder.getDimension()
    );
    const retriever = new Retriever(config.rootDir, embedder, vectorStore);

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
  .option('--dir <dir>', 'Root directory (or set INDEX_DIR env var)', DEFAULT_DIR)
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
  .option('--dir <dir>', 'Root directory (or set INDEX_DIR env var)', DEFAULT_DIR)
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

// ---- REINDEX command ----
program
  .command('reindex')
  .description('Full rebuild: delete collection & cache, recompile TypeScript, then re-index from scratch')
  .argument('[dir]', 'Directory to index (or set INDEX_DIR env var)', DEFAULT_DIR)
  .option('--extensions <exts>', 'Comma-separated file extensions', '.ts,.tsx')
  .option('--qdrant-url <url>', 'Qdrant server URL', DEFAULT_CONFIG.qdrantUrl)
  .option('--collection <name>', 'Qdrant collection name', DEFAULT_CONFIG.collectionName)
  .option('--model <name>', 'Embedding model name', DEFAULT_CONFIG.embeddingModel)
  .option('--openai-key <key>', 'OpenAI API key (or set OPENAI_API_KEY env var)')
  .option('--skip-build', 'Skip TypeScript compilation step', false)
  .action(async (dir: string, opts: Record<string, string | boolean>) => {
    const config = resolveConfig(dir, {
      extensions: typeof opts.extensions === 'string'
        ? opts.extensions.split(',').map((e) => (e.startsWith('.') ? e : `.${e}`))
        : undefined,
      qdrantUrl: opts.qdrantUrl as string | undefined,
      collectionName: opts.collection as string | undefined,
      embeddingModel: opts.model as string | undefined,
      openaiApiKey: opts.openaiKey as string | undefined,
    });

    console.log(chalk.bold.cyan('\n🔄 Full Reindex\n'));
    console.log(chalk.gray(`  Root:       ${config.rootDir}`));
    console.log(chalk.gray(`  Collection: ${config.collectionName}`));
    console.log(chalk.gray(`  Model:      ${config.openaiApiKey ? 'OpenAI text-embedding-3-small (→ local fallback)' : config.embeddingModel}`));
    console.log('');

    const spinner = ora();

    try {
      // Step 1: Build TypeScript (optional)
      if (!opts.skipBuild) {
        spinner.start('Compiling TypeScript...');
        const { execSync } = await import('node:child_process');
        try {
          execSync('npm run build:ts', {
            cwd: path.resolve(import.meta.dirname ?? path.dirname(new URL(import.meta.url).pathname), '..'),
            stdio: 'pipe',
          });
          spinner.succeed('TypeScript compiled');
        } catch (buildErr) {
          spinner.fail('TypeScript compilation failed');
          console.error(chalk.red(`${buildErr}`));
          process.exit(1);
        }
      }

      // Step 2: Delete Qdrant collection
      spinner.start(`Deleting Qdrant collection "${config.collectionName}"...`);
      const indexer = new Indexer(config);
      await indexer.reset();
      spinner.succeed('Collection and cache cleared');

      // Step 3: Initialize and re-index
      spinner.start('Initializing embedding model...');
      await indexer.init();

      const stats = await indexer.index((stage, detail) => {
        spinner.text = `[${stage}] ${detail}`;
      });

      spinner.succeed(chalk.green('Reindex complete!'));
      console.log('');
      console.log(chalk.bold('  📊 Statistics:'));
      console.log(chalk.gray(`     Files:          ${stats.totalFiles}`));
      console.log(chalk.gray(`     Total chunks:   ${stats.totalChunks}`));
      console.log(chalk.gray(`     New embeddings: ${stats.newChunks}`));
      console.log(chalk.gray(`     Time:           ${stats.indexTimeMs}ms`));
      console.log('');
    } catch (err) {
      spinner.fail(chalk.red(`Reindex failed: ${err}`));
      process.exit(1);
    }
  });

// ---- Helpers ----

/** Map file extensions to highlight.js language names */
const EXT_TO_LANG: Record<string, string> = {
  '.ts': 'typescript',
  '.tsx': 'typescript',
  '.js': 'javascript',
  '.jsx': 'javascript',
  '.py': 'python',
  '.rs': 'rust',
  '.go': 'go',
  '.java': 'java',
  '.c': 'c',
  '.cpp': 'cpp',
  '.h': 'c',
  '.hpp': 'cpp',
  '.rb': 'ruby',
  '.swift': 'swift',
  '.kt': 'kotlin',
  '.scala': 'scala',
  '.cs': 'csharp',
  '.php': 'php',
  '.sh': 'bash',
  '.bash': 'bash',
  '.zsh': 'bash',
  '.json': 'json',
  '.yaml': 'yaml',
  '.yml': 'yaml',
  '.md': 'markdown',
  '.sql': 'sql',
  '.html': 'xml',
  '.xml': 'xml',
  '.css': 'css',
  '.scss': 'scss',
  '.less': 'less',
  '.vue': 'typescript',
  '.svelte': 'typescript',
};

function detectLanguage(filePath: string): string | undefined {
  const ext = path.extname(filePath).toLowerCase();
  return EXT_TO_LANG[ext];
}

function highlightCode(code: string, filePath: string): string {
  const lang = detectLanguage(filePath);
  try {
    return highlight(code, { language: lang, ignoreIllegals: true });
  } catch {
    // If highlighting fails, return plain code
    return code;
  }
}

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

    // Syntax-highlight the code block, then add line numbers
    const highlighted = highlightCode(r.content, r.filePath);
    const lines = highlighted.split('\n');
    for (let j = 0; j < lines.length; j++) {
      const lineNum = (r.startLine + j).toString().padStart(4, ' ');
      console.log(chalk.gray(`  ${lineNum} │ `) + lines[j]);
    }
    console.log('');
  }
}

program.parse();
