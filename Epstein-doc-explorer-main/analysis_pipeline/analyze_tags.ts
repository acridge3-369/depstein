#!/usr/bin/env node

import Database from 'better-sqlite3';

const db = new Database('document_analysis.db', { readonly: true });

// Get all tags from all triples
const triples = db.prepare('SELECT triple_tags FROM rdf_triples WHERE triple_tags IS NOT NULL').all() as { triple_tags: string }[];

const tagCounts = new Map<string, number>();

triples.forEach(({ triple_tags }) => {
  try {
    const tags = JSON.parse(triple_tags) as string[];
    tags.forEach(tag => {
      tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
    });
  } catch (e) {
    // Skip invalid JSON
  }
});

// Sort by frequency
const sortedTags = Array.from(tagCounts.entries())
  .sort((a, b) => b[1] - a[1]);

console.log(`Total unique tags: ${sortedTags.length}`);
console.log(`\nTop 100 tags by frequency:\n`);

sortedTags.slice(0, 100).forEach(([tag, count]) => {
  console.log(`${tag.padEnd(50)} ${count}`);
});
