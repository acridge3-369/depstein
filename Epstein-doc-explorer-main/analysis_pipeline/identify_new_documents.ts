#!/usr/bin/env node

import Database from 'better-sqlite3';
import fs from 'fs/promises';
import path from 'path';

const CSV_PATH = '/Users/max/Desktop/digital projects/docnetwork/data/EPS_FILES_20K_NOV2026.csv';
const DB_PATH = 'document_analysis.db';
const OUTPUT_DIR = '/Users/max/Desktop/digital projects/docnetwork/data/new_docs_nov2024';
const TRANCHE_SIZE = 1000;

interface CsvRow {
  filename: string;
  text: string;
}

/**
 * Extract document ID from filename
 * Example: "IMAGES-005-HOUSE_OVERSIGHT_020367.txt" -> "HOUSE_OVERSIGHT_020367"
 */
function extractDocId(filename: string): string {
  // Remove file extension
  const nameWithoutExt = filename.replace('.txt', '');

  // Extract the document ID (everything after the last dash or the whole name)
  const parts = nameWithoutExt.split('-');

  // If it has the IMAGES-XXX- prefix, get everything after that
  if (parts.length >= 3 && parts[0] === 'IMAGES') {
    return parts.slice(2).join('-');
  }

  // Otherwise just remove .txt
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
  let headerParsed = false;

  const lines = content.split('\n');

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Skip header
    if (i === 0) {
      continue;
    }

    // Parse line character by character to handle quoted fields with newlines
    for (let j = 0; j < line.length; j++) {
      const char = line[j];

      if (char === '"') {
        insideQuotes = !insideQuotes;
      } else if (char === ',' && !insideQuotes) {
        // Field separator
        if (!currentRow.filename) {
          currentRow.filename = currentField;
          currentField = '';
        }
      } else {
        currentField += char;
      }
    }

    // If we're not inside quotes, this row is complete
    if (!insideQuotes && currentField) {
      currentRow.text = currentField;

      if (currentRow.filename && currentRow.text) {
        rows.push(currentRow as CsvRow);
      }

      currentRow = {};
      currentField = '';
    } else if (insideQuotes) {
      // Add newline if we're inside quotes (multi-line field)
      currentField += '\n';
    }
  }

  return rows;
}

async function main() {
  console.log('🔍 Identifying New Documents\n');
  console.log('=' . repeat(60));

  // Load existing document IDs from database
  console.log('\n📊 Loading existing documents from database...');
  const db = new Database(DB_PATH, { readonly: true });

  const existingDocs = db.prepare('SELECT doc_id FROM documents').all() as { doc_id: string }[];
  const existingDocIds = new Set(existingDocs.map(d => d.doc_id));

  console.log(`   Found ${existingDocIds.size} existing documents`);

  // Also check for base document IDs (without _partX suffix)
  const baseDocIds = new Set<string>();
  for (const docId of existingDocIds) {
    const baseId = docId.replace(/_part\d+$/, '');
    baseDocIds.add(baseId);
  }

  db.close();

  // Parse CSV file
  console.log('\n📄 Parsing CSV file...');
  const csvRows = await parseCsv(CSV_PATH);
  console.log(`   Found ${csvRows.length} documents in CSV`);

  // Identify new documents
  console.log('\n🆕 Identifying new documents...');
  const newDocs: CsvRow[] = [];
  const duplicates: string[] = [];

  for (const row of csvRows) {
    const docId = extractDocId(row.filename);
    const baseId = docId.replace(/_part\d+$/, '');

    if (existingDocIds.has(docId) || baseDocIds.has(baseId)) {
      duplicates.push(docId);
    } else {
      newDocs.push(row);
    }
  }

  console.log(`   New documents: ${newDocs.length}`);
  console.log(`   Duplicates (skipped): ${duplicates.length}`);

  // Show sample duplicates
  if (duplicates.length > 0) {
    console.log('\n   Sample duplicates:');
    duplicates.slice(0, 10).forEach(id => console.log(`     - ${id}`));
    if (duplicates.length > 10) {
      console.log(`     ... and ${duplicates.length - 10} more`);
    }
  }

  // Create tranches
  console.log(`\n📦 Creating tranches of ${TRANCHE_SIZE} documents each...`);

  await fs.mkdir(OUTPUT_DIR, { recursive: true });

  const numTranches = Math.ceil(newDocs.length / TRANCHE_SIZE);
  console.log(`   Total tranches: ${numTranches}`);

  for (let i = 0; i < numTranches; i++) {
    const start = i * TRANCHE_SIZE;
    const end = Math.min(start + TRANCHE_SIZE, newDocs.length);
    const tranche = newDocs.slice(start, end);

    const tranchePath = path.join(OUTPUT_DIR, `tranche_${String(i + 1).padStart(3, '0')}.csv`);

    // Write CSV
    const csvContent = [
      'filename,text',
      ...tranche.map(row => `${row.filename},"${row.text.replace(/"/g, '""')}"`)
    ].join('\n');

    await fs.writeFile(tranchePath, csvContent);

    console.log(`   ✓ Tranche ${i + 1}/${numTranches}: ${tranche.length} documents -> ${path.basename(tranchePath)}`);
  }

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('✅ Complete!\n');
  console.log(`Summary:`);
  console.log(`  Total CSV documents: ${csvRows.length}`);
  console.log(`  Existing documents: ${duplicates.length}`);
  console.log(`  New documents: ${newDocs.length}`);
  console.log(`  Tranches created: ${numTranches}`);
  console.log(`  Output directory: ${OUTPUT_DIR}`);
  console.log('='.repeat(60));
}

main().catch(console.error);
