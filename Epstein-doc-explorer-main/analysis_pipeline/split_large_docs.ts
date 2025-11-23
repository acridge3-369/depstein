#!/usr/bin/env node

import * as fs from 'fs/promises';
import * as path from 'path';

const MAX_CHARS = 100000; // ~25k tokens worth of text

interface SplitStats {
  totalFiles: number;
  largeFiles: number;
  partsCreated: number;
  largestOriginal: number;
}

/**
 * Split a large document into multiple parts
 */
async function splitDocument(
  inputPath: string,
  outputDir: string,
  docId: string
): Promise<number> {
  const content = await fs.readFile(inputPath, 'utf-8');

  if (content.length <= MAX_CHARS) {
    return 0; // No split needed
  }

  console.log(`Splitting ${docId}: ${content.length} chars`);

  const parts: string[] = [];
  let remaining = content;
  let partNum = 1;

  while (remaining.length > 0) {
    if (remaining.length <= MAX_CHARS) {
      parts.push(remaining);
      break;
    }

    // Try to split at a paragraph or line break near the limit
    let splitPoint = MAX_CHARS;

    // Look backwards for a good split point (paragraph break)
    const searchStart = Math.max(0, MAX_CHARS - 1000);
    const searchText = remaining.substring(searchStart, MAX_CHARS);

    const paragraphBreak = searchText.lastIndexOf('\n\n');
    if (paragraphBreak !== -1) {
      splitPoint = searchStart + paragraphBreak + 2;
    } else {
      // Look for any line break
      const lineBreak = searchText.lastIndexOf('\n');
      if (lineBreak !== -1) {
        splitPoint = searchStart + lineBreak + 1;
      }
    }

    parts.push(remaining.substring(0, splitPoint));
    remaining = remaining.substring(splitPoint);
    partNum++;
  }

  // Write out the parts
  for (let i = 0; i < parts.length; i++) {
    const partId = `${docId}_part${i + 1}`;
    const outputPath = path.join(outputDir, `${partId}.txt`);
    await fs.writeFile(outputPath, parts[i], 'utf-8');
    console.log(`  ✓ Created ${partId}.txt (${parts[i].length} chars)`);
  }

  return parts.length;
}

/**
 * Main function
 */
async function main() {
  const inputDir = process.argv[2] || 'data/001';
  const outputDir = process.argv[3] || 'data/001_split';

  console.log(`\n=== Splitting Large Documents ===\n`);
  console.log(`Input directory: ${inputDir}`);
  console.log(`Output directory: ${outputDir}`);
  console.log(`Max size: ${MAX_CHARS} characters\n`);

  // Create output directory
  await fs.mkdir(outputDir, { recursive: true });

  // Read all text files
  const files = await fs.readdir(inputDir);
  const textFiles = files.filter(f => f.endsWith('.txt'));

  const stats: SplitStats = {
    totalFiles: textFiles.length,
    largeFiles: 0,
    partsCreated: 0,
    largestOriginal: 0
  };

  // Process each file
  for (const file of textFiles) {
    const inputPath = path.join(inputDir, file);
    const docId = file.replace('.txt', '');

    try {
      const fileStats = await fs.stat(inputPath);
      const content = await fs.readFile(inputPath, 'utf-8');

      if (content.length > stats.largestOriginal) {
        stats.largestOriginal = content.length;
      }

      if (content.length > MAX_CHARS) {
        stats.largeFiles++;
        const partsCreated = await splitDocument(inputPath, outputDir, docId);
        stats.partsCreated += partsCreated;
      } else {
        // Copy small files as-is
        const outputPath = path.join(outputDir, file);
        await fs.copyFile(inputPath, outputPath);
      }
    } catch (error) {
      console.error(`Error processing ${file}:`, error);
    }
  }

  console.log(`\n=== Split Complete ===\n`);
  console.log(`Total files: ${stats.totalFiles}`);
  console.log(`Large files split: ${stats.largeFiles}`);
  console.log(`Parts created: ${stats.partsCreated}`);
  console.log(`Largest original: ${stats.largestOriginal.toLocaleString()} characters`);
  console.log(`\nNew files in ${outputDir}:`);

  const outputFiles = await fs.readdir(outputDir);
  console.log(`  Total: ${outputFiles.length} files`);
  console.log(`  Added: ${outputFiles.length - stats.totalFiles + stats.largeFiles} new part files`);
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
