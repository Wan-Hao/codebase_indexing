#!/usr/bin/env node

/**
 * Benchmark CLI — evaluate codebase semantic search quality.
 *
 * Commands:
 *   list                     List supported benchmark datasets
 *   download <dataset>       Download a dataset from HuggingFace
 *   run <dataset>            Run benchmark evaluation
 *
 * Examples:
 *   npx tsx src/cli.ts list
 *   npx tsx src/cli.ts download codesearchnet-python
 *   npx tsx src/cli.ts run codesearchnet-python --model Xenova/bge-base-en-v1.5
 *   npx tsx src/cli.ts run codesearchnet-python --max-corpus 5000 --max-queries 500
 */

import fs from 'node:fs';
import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import { DatasetLoader } from './dataset-loader.js';
import { BenchmarkRunner } from './runner.js';
import { SUPPORTED_DATASETS } from './types.js';
import type { BenchmarkResult } from './types.js';

const program = new Command();

program
  .name('bench')
  .description('Benchmark evaluation for codebase semantic search')
  .version('0.1.0');

// ---- LIST command ----
program
  .command('list')
  .description('List supported benchmark datasets')
  .action(() => {
    console.log(chalk.bold.cyan('\n📋 Supported Benchmark Datasets\n'));

    const loader = new DatasetLoader();
    for (const ds of SUPPORTED_DATASETS) {
      const cached = loader.isCached(ds.id) ? chalk.green(' [cached]') : '';
      console.log(`  ${chalk.bold(ds.id)}${cached}`);
      console.log(chalk.gray(`    ${ds.description}`));
      console.log(chalk.gray(`    HuggingFace: ${ds.hfRepo}`));
      console.log('');
    }
  });

// ---- DOWNLOAD command ----
program
  .command('download')
  .description('Download a benchmark dataset from HuggingFace')
  .argument('<dataset>', 'Dataset ID (e.g. codesearchnet-python)')
  .option('--cache-dir <dir>', 'Cache directory', '.bench-cache')
  .action(async (datasetId: string, opts: { cacheDir: string }) => {
    console.log(chalk.bold.cyan(`\n📥 Downloading: ${datasetId}\n`));

    const loader = new DatasetLoader(opts.cacheDir);
    const info = loader.getDatasetInfo(datasetId);
    if (!info) {
      console.error(
        chalk.red(`Unknown dataset: ${datasetId}\n`) +
          chalk.gray(
            `Available: ${SUPPORTED_DATASETS.map((d) => d.id).join(', ')}`
          )
      );
      process.exit(1);
    }

    if (loader.isCached(datasetId)) {
      console.log(chalk.yellow('  Dataset already cached. Re-downloading...\n'));
    }

    const spinner = ora('Starting download...').start();

    try {
      await loader.download(datasetId, (stage, detail) => {
        spinner.text = `[${stage}] ${detail}`;
      });
      spinner.succeed(chalk.green('Download complete!'));
    } catch (err) {
      spinner.fail(chalk.red(`Download failed: ${err}`));
      process.exit(1);
    }
  });

// ---- RUN command ----
program
  .command('run')
  .description('Run benchmark evaluation on a dataset')
  .argument('<dataset>', 'Dataset ID (e.g. codesearchnet-python)')
  .option('--model <name>', 'Embedding model name', 'Xenova/bge-base-en-v1.5')
  .option('--max-corpus <n>', 'Max corpus entries (for quick testing)', parseInt)
  .option('--max-queries <n>', 'Max queries (for quick testing)', parseInt)
  .option('--batch-size <n>', 'Embedding batch size', parseInt)
  .option('--cache-dir <dir>', 'Cache directory', '.bench-cache')
  .action(
    async (
      datasetId: string,
      opts: {
        model: string;
        maxCorpus?: number;
        maxQueries?: number;
        batchSize?: number;
        cacheDir: string;
      }
    ) => {
      console.log(chalk.bold.cyan('\n🏋️  Benchmark Runner\n'));
      console.log(chalk.gray(`  Dataset:     ${datasetId}`));
      console.log(chalk.gray(`  Model:       ${opts.model}`));
      if (opts.maxCorpus) {
        console.log(chalk.gray(`  Max corpus:  ${opts.maxCorpus}`));
      }
      if (opts.maxQueries) {
        console.log(chalk.gray(`  Max queries: ${opts.maxQueries}`));
      }
      console.log('');

      const runner = new BenchmarkRunner(opts.cacheDir);
      const spinner = ora('Initializing...').start();

      try {
        const result = await runner.run(
          datasetId,
          {
            model: opts.model,
            maxCorpus: opts.maxCorpus,
            maxQueries: opts.maxQueries,
            batchSize: opts.batchSize,
            cacheDir: opts.cacheDir,
          },
          (stage, detail) => {
            spinner.text = `[${stage}] ${detail}`;
          }
        );

        spinner.succeed(chalk.green('Benchmark complete!'));
        printResults(result);
      } catch (err) {
        spinner.fail(chalk.red(`Benchmark failed: ${err}`));
        process.exit(1);
      }
    }
  );

// ---- Result Printing ----

function printResults(result: BenchmarkResult) {
  console.log(chalk.bold.cyan('\n📊 Results\n'));
  console.log(chalk.gray(`  Dataset:  ${result.dataset}`));
  console.log(chalk.gray(`  Model:    ${result.model}`));
  console.log(chalk.gray(`  Queries:  ${result.numQueries}`));
  console.log(chalk.gray(`  Corpus:   ${result.numCorpus}`));
  console.log('');

  // Metrics table
  console.log(chalk.bold('  Retrieval Metrics:'));
  console.log(chalk.gray('  ' + '─'.repeat(42)));

  const m = result.metrics;
  const rows: Array<[string, number]> = [
    ['MRR@1', m.mrr_at_1],
    ['MRR@5', m.mrr_at_5],
    ['MRR@10', m.mrr_at_10],
    ['NDCG@1', m.ndcg_at_1],
    ['NDCG@5', m.ndcg_at_5],
    ['NDCG@10', m.ndcg_at_10],
    ['Recall@1', m.recall_at_1],
    ['Recall@5', m.recall_at_5],
    ['Recall@10', m.recall_at_10],
    ['Recall@100', m.recall_at_100],
  ];

  for (const [name, value] of rows) {
    const pct = (value * 100).toFixed(2);
    const bar = makeBar(value, 20);
    const color = value > 0.5 ? chalk.green : value > 0.2 ? chalk.yellow : chalk.red;
    console.log(
      `  ${chalk.white(name.padEnd(12))} ${color(pct.padStart(6))}%  ${chalk.gray(bar)}`
    );
  }

  console.log('');
  console.log(chalk.bold('  Timing:'));
  console.log(chalk.gray('  ' + '─'.repeat(42)));
  console.log(
    chalk.gray(
      `  Embed corpus:   ${(result.timing.embedCorpusMs / 1000).toFixed(1)}s`
    )
  );
  console.log(
    chalk.gray(
      `  Embed queries:  ${(result.timing.embedQueriesMs / 1000).toFixed(1)}s`
    )
  );
  console.log(
    chalk.gray(`  Search:         ${(result.timing.searchMs / 1000).toFixed(1)}s`)
  );
  console.log(
    chalk.gray(`  Total:          ${(result.timing.totalMs / 1000).toFixed(1)}s`)
  );
  console.log('');

  // Save result to file
  const resultPath = `.bench-cache/results/${result.dataset}_${modelToPathSafe(result.model)}.json`;
  const dir = resultPath.substring(0, resultPath.lastIndexOf('/'));
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(resultPath, JSON.stringify(result, null, 2));
  console.log(chalk.gray(`  Results saved to: ${resultPath}\n`));
}

function makeBar(value: number, width: number): string {
  const filled = Math.round(value * width);
  return '█'.repeat(filled) + '░'.repeat(width - filled);
}

function modelToPathSafe(model: string): string {
  return model.replace(/[\/\\:]/g, '_');
}

program.parse();
