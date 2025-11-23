#!/usr/bin/env node

import Database from 'better-sqlite3';

const db = new Database('document_analysis.db');

console.log('🔧 Fixing unknown/anonymous actor names by document...\n');

// Patterns to detect unknown/anonymous actors
const unknownPatterns = [
  /unknown\s+person/i,
  /unknown\s+individual/i,
  /unknown\s+man/i,
  /unknown\s+woman/i,
  /jane\s+doe/i,
  /john\s+doe/i,
  /\bdoe\s+\d+/i,  // Doe 1, Doe 2, etc.
  /unidentified/i,
  /anonymous/i,
];

function isUnknownActor(name: string): boolean {
  return unknownPatterns.some(pattern => pattern.test(name));
}

// Get all triples with unknown actors
const triples = db.prepare(`
  SELECT id, doc_id, actor, target
  FROM rdf_triples
`).all() as Array<{
  id: number;
  doc_id: string;
  actor: string;
  target: string;
}>;

console.log(`Total triples: ${triples.length}`);

// Track updates
const updates: Array<{
  id: number;
  newActor: string | null;
  newTarget: string | null;
}> = [];

let actorUpdates = 0;
let targetUpdates = 0;

for (const triple of triples) {
  let newActor: string | null = null;
  let newTarget: string | null = null;

  // Check if actor is unknown
  if (isUnknownActor(triple.actor)) {
    // Append document ID to make it unique
    newActor = `${triple.actor} (${triple.doc_id})`;
    actorUpdates++;
  }

  // Check if target is unknown
  if (isUnknownActor(triple.target)) {
    // Append document ID to make it unique
    newTarget = `${triple.target} (${triple.doc_id})`;
    targetUpdates++;
  }

  if (newActor !== null || newTarget !== null) {
    updates.push({
      id: triple.id,
      newActor,
      newTarget
    });
  }
}

console.log(`\nFound ${updates.length} triples to update`);
console.log(`  Actor updates: ${actorUpdates}`);
console.log(`  Target updates: ${targetUpdates}`);

if (updates.length === 0) {
  console.log('\n✅ No updates needed!');
  db.close();
  process.exit(0);
}

// Apply updates in a transaction
console.log('\n📝 Applying updates...');

const updateStmt = db.prepare(`
  UPDATE rdf_triples
  SET
    actor = COALESCE(?, actor),
    target = COALESCE(?, target)
  WHERE id = ?
`);

const applyUpdates = db.transaction((updates: typeof updates) => {
  for (const update of updates) {
    updateStmt.run(
      update.newActor,
      update.newTarget,
      update.id
    );
  }
});

applyUpdates(updates);

console.log(`✅ Updated ${updates.length} triples`);

// Verify results
console.log('\n🔍 Verifying updates...');

const verifyActors = db.prepare(`
  SELECT DISTINCT actor
  FROM rdf_triples
  WHERE actor LIKE '%(%)'
`).all() as { actor: string }[];

const verifyTargets = db.prepare(`
  SELECT DISTINCT target
  FROM rdf_triples
  WHERE target LIKE '%(%)'
`).all() as { target: string }[];

console.log(`\nActors with document IDs: ${verifyActors.length}`);
console.log(`Targets with document IDs: ${verifyTargets.length}`);

// Show some examples
console.log('\nExample updated actors:');
for (const { actor } of verifyActors.slice(0, 10)) {
  console.log(`  - ${actor}`);
}

db.close();

console.log('\n✅ Fix complete!');
