#!/usr/bin/env node

import Database from 'better-sqlite3';

console.log('📊 Adding tag_embeddings table...');

const db = new Database('document_analysis.db');

// Create tag_embeddings table
db.exec(`
  CREATE TABLE IF NOT EXISTS tag_embeddings (
    tag TEXT PRIMARY KEY,
    embedding TEXT NOT NULL,  -- JSON array of 32 floats
    model TEXT NOT NULL,       -- Model used (e.g., "Qwen3-Embedding-0.6B-ONNX-fp16-32d")
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
  );
`);

console.log('✅ Created tag_embeddings table');

// Check if any embeddings exist
const count = db.prepare('SELECT COUNT(*) as count FROM tag_embeddings').get() as { count: number };
console.log(`📊 Current embeddings in database: ${count.count}`);

db.close();

console.log('✅ Migration complete!');
