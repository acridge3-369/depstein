#!/usr/bin/env node

import * as fs from 'fs/promises';
import * as path from 'path';

const MAX_CHARS = 100000; // ~25k tokens worth of text
const TRANCHES_DIR = '/Users/max/Desktop/digital projects/docnetwork/data/new_docs_nov2024';
const OUTPUT_BASE = '/Users/max/Desktop/digital projects/docnetwork/data/new_docs_nov2024_processed';

interface CsvRow {
  filename: string;
  text: string;
}

/**
 * Extract document ID from filename
 */
function extractDocId(filename: string): string {
  const nameWithoutExt = filename.replace('.txt', '');
  const parts = nameWithoutExt.split('-');
  if (parts.length >= 3 && parts[0] === 'IMAGES') {
    return parts.slice(2).join('-');
  }
  return nameWithoutExt;
}

/**
 * Parse CSV file (handling multi-line text fields)
 */
async function parseCsv(filePath: string): Promise<CsvRow[]> {
  const content = await fs.readFile(filePath, 'utf-8');
  const rows: CsvRow[] = [];
  let currentRow: Partial<CsvRow> = {};
  let currentField = '';
  let insideQuotes = false;
  const lines = content.split('\n');

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (i === 0) continue; // Skip header

    for (let j = 0; j < line.length; j++) {
      const char = line[j];
      if (char === '"') {
        insideQuotes = !insideQuotes;
      } else if (char === ',' && !insideQuotes) {
        if (!currentRow.filename) {
          currentRow.filename = currentField;
          currentField = '';
        }
      } else {
        currentField += char;
      }
    }

    if (!insideQuotes && currentField) {
      currentRow.text = currentField;
      if (currentRow.filename && currentRow.text) {
        rows.push(currentRow as CsvRow);
      }
      currentRow = {};
      currentField = '';
    } else if (insideQuotes) {
      currentField += '\n';
    }
  }

  return rows;
}

/**
 * Split a large document into multiple parts
 */
function splitText(text: string, docId: string): { parts: string[], partIds: string[] } {
  if (text.length <= MAX_CHARS) {
    return { parts: [text], partIds: [docId] };
  }

  const parts: string[] = [];
  const partIds: string[] = [];
  let remaining = text;
  let partNum = 1;

  while (remaining.length > 0) {
    if (remaining.length <= MAX_CHARS) {
      parts.push(remaining);
      partIds.push(`${docId}_part${partNum}`);
      break;
    }

    let splitPoint = MAX_CHARS;
    const searchStart = Math.max(0, MAX_CHARS - 1000);
    const searchText = remaining.substring(searchStart, MAX_CHARS);

    const paragraphBreak = searchText.lastIndexOf('\n\n');
    if (paragraphBreak !== -1) {
      splitPoint = searchStart + paragraphBreak + 2;
    } else {
      const lineBreak = searchText.lastIndexOf('\n');
      if (lineBreak !== -1) {
        splitPoint = searchStart + lineBreak + 1;
      }
    }

    parts.push(remaining.substring(0, splitPoint));
    partIds.push(`${docId}_part${partNum}`);
    remaining = remaining.substring(splitPoint);
    partNum++;
  }

  return { parts, partIds };
}

/**
 * Process a single tranche
 */
async function processTranche(trancheNum: number): Promise<void> {
  const trancheFile = path.join(TRANCHES_DIR, `tranche_${String(trancheNum).padStart(3, '0')}.csv`);
  const outputDir = path.join(OUTPUT_BASE, `tranche_${String(trancheNum).padStart(3, '0')}`);

  console.log(`\n📦 Processing Tranche ${trancheNum}`);
  console.log(`   Input: ${trancheFile}`);
  console.log(`   Output: ${outputDir}`);

  // Create output directory
  await fs.mkdir(outputDir, { recursive: true });

  // Parse CSV
  const rows = await parseCsv(trancheFile);
  console.log(`   Documents in CSV: ${rows.length}`);

  let totalDocs = 0;
  let splitDocs = 0;
  let totalParts = 0;

  // Process each document
  for (const row of rows) {
    const docId = extractDocId(row.filename);
    const { parts, partIds } = splitText(row.text, docId);

    if (parts.length > 1) {
      splitDocs++;
      console.log(`   Split ${docId}: ${parts.length} parts (${row.text.length} chars)`);
    }

    totalDocs++;
    totalParts += parts.length;

    // Write parts
    for (let i = 0; i < parts.length; i++) {
      const outputPath = path.join(outputDir, `${partIds[i]}.txt`);
      await fs.writeFile(outputPath, parts[i], 'utf-8');
    }
  }

  console.log(`   ✓ Processed ${totalDocs} documents`);
  console.log(`   ✓ Split ${splitDocs} large documents`);
  console.log(`   ✓ Created ${totalParts} total files`);
}

/**
 * Main function
 */
async function main() {
  const trancheArg = process.argv[2];

  console.log('🔄 Processing CSV Tranches to Text Files\n');
  console.log('='.repeat(60));

  if (trancheArg === 'all') {
    // Process all 24 tranches
    console.log('Processing all 24 tranches...\n');

    for (let i = 1; i <= 24; i++) {
      try {
        await processTranche(i);
      } catch (error) {
        console.error(`Error processing tranche ${i}:`, error);
      }
    }
  } else if (trancheArg) {
    // Process specific tranche
    const trancheNum = parseInt(trancheArg);
    if (isNaN(trancheNum) || trancheNum < 1 || trancheNum > 24) {
      console.error('Invalid tranche number. Must be 1-24 or "all"');
      process.exit(1);
    }

    await processTranche(trancheNum);
  } else {
    console.log('Usage: npx tsx process_csv_tranches.ts <tranche_number | all>');
    console.log('Examples:');
    console.log('  npx tsx process_csv_tranches.ts 1');
    console.log('  npx tsx process_csv_tranches.ts all');
    process.exit(1);
  }

  console.log('\n' + '='.repeat(60));
  console.log('✅ Complete!');
  console.log('='.repeat(60));
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
