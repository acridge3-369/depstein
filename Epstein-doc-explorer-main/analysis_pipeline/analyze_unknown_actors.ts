#!/usr/bin/env node

import Database from 'better-sqlite3';

const db = new Database('document_analysis.db');

console.log('🔍 Analyzing unknown/anonymous actors...\n');

// Patterns to detect unknown/anonymous actors
const unknownPatterns = [
  /unknown\s+person/i,
  /unknown\s+individual/i,
  /unknown\s+man/i,
  /unknown\s+woman/i,
  /jane\s+doe/i,
  /john\s+doe/i,
  /\bdoe\b/i,
  /unidentified/i,
  /anonymous/i,
];

// Get all unique actors and targets
const actors = db.prepare(`
  SELECT DISTINCT actor as name FROM rdf_triples
  UNION
  SELECT DISTINCT target as name FROM rdf_triples
`).all() as { name: string }[];

console.log(`Total unique actors/targets: ${actors.length}\n`);

// Find matches for each pattern
const matchesByPattern = new Map<string, Set<string>>();

for (const pattern of unknownPatterns) {
  const matches = new Set<string>();

  for (const { name } of actors) {
    if (pattern.test(name)) {
      matches.add(name);
    }
  }

  if (matches.size > 0) {
    matchesByPattern.set(pattern.source, matches);
  }
}

// Display results
console.log('📊 Matches by pattern:\n');

for (const [pattern, matches] of matchesByPattern.entries()) {
  console.log(`Pattern: /${pattern}/i`);
  console.log(`  Matches: ${matches.size}`);
  console.log(`  Examples: ${Array.from(matches).slice(0, 5).join(', ')}`);
  console.log();
}

// Get all unique unknown/anonymous names
const allUnknownNames = new Set<string>();
for (const matches of matchesByPattern.values()) {
  for (const match of matches) {
    allUnknownNames.add(match);
  }
}

console.log(`\n📋 Total unique unknown/anonymous names: ${allUnknownNames.size}\n`);

// For each unknown name, count how many documents it appears in
console.log('📄 Document distribution for unknown actors:\n');

const nameDocCounts = new Map<string, { docCount: number; tripleCount: number }>();

for (const name of allUnknownNames) {
  const result = db.prepare(`
    SELECT
      COUNT(DISTINCT doc_id) as doc_count,
      COUNT(*) as triple_count
    FROM rdf_triples
    WHERE actor = ? OR target = ?
  `).get(name, name) as { doc_count: number; triple_count: number };

  nameDocCounts.set(name, {
    docCount: result.doc_count,
    tripleCount: result.triple_count
  });
}

// Sort by document count (descending) and show top offenders
const sorted = Array.from(nameDocCounts.entries())
  .sort((a, b) => b[1].docCount - a[1].docCount);

console.log('Top 20 unknown actors by document spread:\n');
for (const [name, counts] of sorted.slice(0, 20)) {
  console.log(`  "${name}"`);
  console.log(`    Appears in ${counts.docCount} documents`);
  console.log(`    Total relationships: ${counts.tripleCount}`);
  console.log();
}

// Calculate total impact
const totalTriples = db.prepare('SELECT COUNT(*) as count FROM rdf_triples').get() as { count: number };
const unknownTriples = sorted.reduce((sum, [_, counts]) => sum + counts.tripleCount, 0);

console.log('\n📈 Impact summary:');
console.log(`  Total triples: ${totalTriples.count}`);
console.log(`  Triples with unknown actors: ${unknownTriples}`);
console.log(`  Percentage: ${((unknownTriples / totalTriples.count) * 100).toFixed(2)}%`);

db.close();
