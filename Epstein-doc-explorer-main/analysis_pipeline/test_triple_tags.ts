#!/usr/bin/env node

import { query } from '@anthropic-ai/claude-agent-sdk';
import * as fs from 'fs/promises';
import Database from 'better-sqlite3';

const ANALYSIS_MODEL = 'claude-haiku-4-5';
const db = new Database('document_analysis.db');

// First, let's check the schema and add the triple_tags column if needed
console.log('=== Checking database schema ===\n');

const schemaInfo = db.prepare('PRAGMA table_info(rdf_triples)').all();
console.log('Current rdf_triples schema:', schemaInfo);

const hasTripleTags = schemaInfo.some((col: any) => col.name === 'triple_tags');
const hasExplicitTopic = schemaInfo.some((col: any) => col.name === 'explicit_topic');
const hasImplicitTopic = schemaInfo.some((col: any) => col.name === 'implicit_topic');
const hasActorLikelyType = schemaInfo.some((col: any) => col.name === 'actor_likely_type');

if (!hasTripleTags) {
  console.log('\nAdding triple_tags column to rdf_triples table...');
  db.exec('ALTER TABLE rdf_triples ADD COLUMN triple_tags TEXT');
  console.log('✓ Column added');
} else {
  console.log('\n✓ triple_tags column already exists');
}

if (!hasExplicitTopic) {
  console.log('Adding explicit_topic column to rdf_triples table...');
  db.exec('ALTER TABLE rdf_triples ADD COLUMN explicit_topic TEXT');
  console.log('✓ Column added');
} else {
  console.log('✓ explicit_topic column already exists');
}

if (!hasImplicitTopic) {
  console.log('Adding implicit_topic column to rdf_triples table...');
  db.exec('ALTER TABLE rdf_triples ADD COLUMN implicit_topic TEXT');
  console.log('✓ Column added');
} else {
  console.log('✓ implicit_topic column already exists');
}

if (!hasActorLikelyType) {
  console.log('Adding actor_likely_type column to rdf_triples table...');
  db.exec('ALTER TABLE rdf_triples ADD COLUMN actor_likely_type TEXT');
  console.log('✓ Column added\n');
} else {
  console.log('✓ actor_likely_type column already exists\n');
}

// Get a test document with unknown actors
const testDocs = db.prepare(`
  SELECT DISTINCT rt.doc_id, d.file_path
  FROM rdf_triples rt
  JOIN documents d ON rt.doc_id = d.doc_id
  WHERE rt.actor LIKE '%unknown%'
  LIMIT 1
`).all() as Array<{ doc_id: string; file_path: string }>;

if (testDocs.length === 0) {
  console.error('No documents found in database');
  process.exit(1);
}

const testDoc = testDocs[0];
console.log(`Testing with document: ${testDoc.doc_id}`);
console.log(`File path: ${testDoc.file_path}\n`);

// Read the document content
const content = await fs.readFile(testDoc.file_path, 'utf-8');

// Enhanced analysis prompt with Epstein aliases AND triple-level tags
const analysisPrompt = `You are analyzing a document from a legal/investigative document collection. The document ID is "${testDoc.doc_id}".

IMPORTANT: You have ALL the information you need in the document text below. Do NOT attempt to read files, explore directories, or gather additional context. Analyze ONLY the text provided.

**CRITICAL IDENTIFICATION RULES:**
This document may contain communications involving Jeffrey Epstein. He may appear under these identifiers:
- Email: jeeitunes@gmail.com
- Email: e:jeeitunes@gmail.com
- Name: jee
- Name: Jeffrey Epstein
- Name: Jeffrey
- Name: Epstein

When you see ANY of these identifiers as a sender, participant, or actor, you MUST use "Jeffrey Epstein" as the actor name in your RDF triples. DO NOT use "jee", "unknown person", or any other placeholder.

Here is the document text:
\`\`\`
${content}
\`\`\`

Your task is to analyze this document and extract structured information. Focus on:

1. **Main actors/participants** - People, organizations, entities mentioned or involved
2. **Key events and actions** - What happened, when, between whom
3. **Temporal information** - Dates, times, sequences of events
4. **Document type and content** - What kind of document is this?
5. **Key themes and topics** - What is this document about?

Return ONLY a valid JSON object with the following structure:

\`\`\`json
{
  "one_sentence_summary": "A brief one-sentence summary including main actors, e.g., 'An email conversation between John Doe and Jane Smith regarding budget approval'",
  "paragraph_summary": "A detailed paragraph (3-5 sentences) explaining the document's content, context, significance, and key points. Include who is involved, what happened, why it matters, and any important outcomes or implications.",
  "date_range_earliest": "YYYY-MM-DD or YYYY-MM-DDTHH:MM format if dates are visible in the document, otherwise null",
  "date_range_latest": "YYYY-MM-DD or YYYY-MM-DDTHH:MM format if dates are visible in the document, otherwise null",
  "category": "One of: court_filing, email, letter, memorandum, report, transcript, financial_document, media_article, book_excerpt, photo_caption, mixed_document, other",
  "content_tags": ["array", "of", "relevant", "document-level", "tags"],
  "rdf_triples": [
    {
      "timestamp": "YYYY-MM-DD or YYYY-MM-DDTHH:MM if available, otherwise omit this field",
      "actor": "PERSON NAME ONLY - Use 'Jeffrey Epstein' when you see jeeitunes@gmail.com or 'jee'",
      "action": "the action verb (e.g., 'sent email to', 'met with', 'testified before', 'paid', 'attended')",
      "target": "PERSON NAME ONLY - not organizations, movies, places (e.g., 'Donald Trump', not 'Donald Trump at party' or '12 Years a Slave')",
      "location": "physical location if mentioned (e.g., 'Mar-a-Lago', 'New York City', 'Palm Beach courthouse'), otherwise omit this field",
      "actor_likely_type": "OPTIONAL - only include if actor is unknown/unnamed/redacted AND there is sufficient evidence to infer their likely type. Type of person - examples include but are not limited to: 'victim', 'witness', 'celebrity', 'political operator', 'staff member', 'law enforcement', 'family member', 'business associate', 'government official'. Use the most specific and appropriate type based on context. Omit entirely if actor is named OR if type cannot be reasonably inferred from context.",
      "tags": ["array", "of", "specific", "tags", "for", "this", "triple", "describing", "the", "nature", "of", "this", "specific", "interaction"],
      "explicit_topic": "short phrase describing the main theme directly evidenced in the surrounding content (e.g., 'biographical facts', 'coordination of business meeting', 'testimony about alleged assault')",
      "implicit_topic": "short phrase describing what the interaction likely relates to, even if not directly stated (e.g., 'relationship cultivation', 'reputation management', 'legal strategy coordination')"
    }
  ]
}
\`\`\`

Guidelines for RDF triples:
- Create a sequential array capturing the key relationships and events in the document
- Include timestamps when dates/times are mentioned in the document
- **CRITICAL - Actor field**: Actor must ALWAYS be a PERSON NAME ONLY
  - ✅ Good: actor: "Jeffrey Epstein" (when you see jeeitunes@gmail.com or jee)
  - ✅ Good: actor: "Donald Trump", "Ghislaine Maxwell"
  - ❌ Bad: actor: "jee" (use "Jeffrey Epstein" instead)
  - ❌ Bad: actor: "FBI" (organization), actor: "United States" (country), actor: "the investigation" (abstract)
  - Only actual human persons can be actors
- **Target field**: Target can be a person, place, organization, or entity
  - ✅ Good: target: "Donald Trump" (person), target: "Hong Kong" (place), target: "FBI" (organization), target: "Mar-a-Lago" (location)
  - ✅ Good: target: "12 Years a Slave" (movie/book), target: "United States Congress" (organization)
  - ❌ Bad: target: "Donald Trump at Mar-a-Lago" (don't combine person with location)
  - If target is a location, ALSO include it in the location field
- **Unknown/Redacted persons**: Use placeholders like "unknown person A", "unknown person B" ONLY when referring to actual unnamed PEOPLE
  - ✅ Good: "unknown person A" for an unnamed victim or redacted individual
  - ❌ Bad: "unknown person A" as placeholder for Jeffrey Epstein when you see jeeitunes@gmail.com or jee
  - **NEW**: When actor is unknown/unnamed/redacted AND you can reasonably infer their type, include "actor_likely_type" field
    - Examples include but are not limited to: "victim", "witness", "celebrity", "political operator", "staff member", "law enforcement", "family member", "business associate", "government official", "legal counsel", "journalist", "minor", "employee", "associate"
    - Choose the most specific and contextually appropriate type that can be reasonably inferred
    - **IMPORTANT**: Only include this field if there is clear contextual evidence. If uncertain or speculative, omit the field entirely rather than guessing
- Use consistent naming (e.g., always "Jeffrey Epstein" not "Epstein" or "Jeffrey" or "jee")
- Actions should be descriptive verb phrases (e.g., "met with", "sent email to", "testified before", "traveled to")
- Focus on person-to-person AND person-to-entity relationships and interactions
- Order triples chronologically when timestamps are available, otherwise by document order
- Extract sufficient triples from each document to accurately capture the nature of relationships and actions documented within

**NEW - Triple-level tags:**
- Each triple should have a "tags" array with specific descriptive tags for THAT INTERACTION
- Tags should describe the nature, context, or category of the specific interaction
- Be specific and descriptive (use snake_case for multi-word tags)
- Examples of triple tags:
  - For "Jeffrey Epstein sent email to Bill Clinton about fundraising": ["email_communication", "political_fundraising", "personal_correspondence"]
  - For "Jane Doe testified before grand jury regarding sexual assault": ["witness_testimony", "sexual_assault_allegations", "legal_proceedings", "grand_jury"]
  - For "Donald Trump met with Vladimir Putin in Helsinki": ["diplomatic_meeting", "international_relations", "summit", "foreign_policy"]
  - For "John Smith transferred $50,000 to offshore account": ["financial_transactions", "offshore_banking", "money_transfer"]
- Triple tags can overlap with or be more specific than document-level tags
- Include 2-6 tags per triple depending on complexity

**NEW - Explicit and Implicit Topics:**
- Each triple must have both an "explicit_topic" and "implicit_topic" field
- Both should be SHORT PHRASES (3-7 words) describing the theme/intent
- **explicit_topic**: What the interaction is DIRECTLY about based on the surrounding text
  - Examples: "biographical facts", "coordination of business meeting", "testimony about alleged assault", "social event attendance", "financial transaction documentation", "legal representation arrangement"
- **implicit_topic**: What the interaction LIKELY relates to or implies, even if not directly stated
  - Examples: "relationship cultivation", "reputation management", "legal strategy coordination", "power networking", "financial concealment", "influence building"
- These help with semantic search and understanding the PURPOSE behind interactions
- Example for "Jeffrey Epstein met with Bill Clinton at Mar-a-Lago":
  - explicit_topic: "social meeting at private club"
  - implicit_topic: "high-level networking and relationship building"
- Example for "Jane Doe testified before grand jury":
  - explicit_topic: "witness testimony in criminal proceedings"
  - implicit_topic: "evidence gathering for prosecution"

Guidelines for document-level content tags:
- Be specific and descriptive (good: "financial_transactions", bad: "money")
- Use snake_case for multi-word tags
- Include both broad topics and specific themes
- Typical tags might include: legal_proceedings, financial_transactions, political_activity, media_coverage, personal_relationships, travel, communications, business_dealings, etc.

If the document is too fragmentary or unreadable to analyze, still provide your best interpretation and mark uncertainty in the summaries.`;

console.log('Running analysis...\n');

let responseText = '';
let usageStats: any = null;
let agentCostUSD = 0;

const agent = query({
  prompt: analysisPrompt,
  options: {
    model: ANALYSIS_MODEL,
    maxTokens: 16000,
    maxTurns: 5,
    allowedTools: [],
  }
});

for await (const message of agent) {
  if (message.type === 'result' && message.subtype === 'success') {
    responseText = message.result;
    if (message.usage) {
      usageStats = message.usage;
    }
    if (message.total_cost_usd !== undefined) {
      agentCostUSD = message.total_cost_usd;
    }
  } else if (message.type === 'assistant') {
    const textBlocks = message.message.content.filter((c: any) => c.type === 'text');
    for (const block of textBlocks) {
      responseText += block.text;
    }
  }
}

// Parse JSON response
const jsonMatch = responseText.match(/```(?:json)?\s*([\s\S]*?)```/) || responseText.match(/\{[\s\S]*\}/);
if (!jsonMatch) {
  console.error('❌ No JSON found in response');
  process.exit(1);
}

const jsonText = jsonMatch[1] || jsonMatch[0];
let analysis;

try {
  analysis = JSON.parse(jsonText);
} catch (parseError) {
  console.log('⚠️  Initial JSON parse failed, attempting repair...\n');

  // Attempt to repair the JSON
  const repairPrompt = `The following JSON response from a document analysis has a parsing error. Please identify and fix the issue, then return ONLY the corrected JSON:

\`\`\`json
${jsonText}
\`\`\`

Parse error: ${parseError instanceof Error ? parseError.message : String(parseError)}

Instructions:
- Find and fix the JSON syntax error (missing quotes, trailing commas, unescaped characters, etc.)
- Ensure all field names are properly quoted
- Ensure all string values are properly escaped
- Return ONLY the valid JSON object, no explanations
- Do NOT modify the content, only fix the syntax`;

  const repairAgent = query({
    prompt: repairPrompt,
    options: {
      model: ANALYSIS_MODEL,
      maxTokens: 16000,
      maxTurns: 3,
      allowedTools: [],
    }
  });

  let repairedText = '';
  for await (const message of repairAgent) {
    if (message.type === 'result' && message.subtype === 'success') {
      repairedText = message.result;
    } else if (message.type === 'assistant') {
      const textBlocks = message.message.content.filter((c: any) => c.type === 'text');
      for (const block of textBlocks) {
        repairedText += block.text;
      }
    }
  }

  // Extract JSON from repair response
  const repairMatch = repairedText.match(/```(?:json)?\s*([\s\S]*?)```/) || repairedText.match(/\{[\s\S]*\}/);
  if (!repairMatch) {
    console.error('❌ Repair failed: No JSON found in repair response');
    console.error('Original error:', parseError);
    process.exit(1);
  }

  const repairedJsonText = repairMatch[1] || repairMatch[0];

  try {
    analysis = JSON.parse(repairedJsonText);
    console.log('✓ JSON successfully repaired!\n');
  } catch (repairError) {
    console.error('❌ Repair failed: Still cannot parse JSON');
    console.error('Original error:', parseError);
    console.error('Repair error:', repairError);
    console.error('\nOriginal JSON:\n', jsonText);
    console.error('\nRepaired JSON:\n', repairedJsonText);
    process.exit(1);
  }
}

console.log('=== Analysis Result ===\n');
console.log('Document Summary:');
console.log(`  ${analysis.one_sentence_summary}\n`);
console.log('Category:', analysis.category);
console.log('Date Range:', analysis.date_range_earliest, 'to', analysis.date_range_latest);
console.log('\nDocument Tags:', analysis.content_tags);
console.log(`\nExtracted ${analysis.rdf_triples.length} triples:\n`);

for (let i = 0; i < analysis.rdf_triples.length; i++) {
  const triple = analysis.rdf_triples[i];
  console.log(`Triple ${i + 1}:`);
  console.log(`  ${triple.timestamp || 'No date'}: ${triple.actor} → ${triple.action} → ${triple.target}`);
  if (triple.actor_likely_type) {
    console.log(`  Actor Type: ${triple.actor_likely_type}`);
  }
  if (triple.location) {
    console.log(`  Location: ${triple.location}`);
  }
  console.log(`  Explicit Topic: ${triple.explicit_topic || 'MISSING!'}`);
  console.log(`  Implicit Topic: ${triple.implicit_topic || 'MISSING!'}`);
  console.log(`  Tags: ${triple.tags ? triple.tags.join(', ') : 'NONE - ERROR!'}`);
  console.log('');
}

console.log('\nUsage Stats:');
console.log(`  Input tokens: ${usageStats?.input_tokens || 0}`);
console.log(`  Output tokens: ${usageStats?.output_tokens || 0}`);
console.log(`  Cost: $${agentCostUSD.toFixed(4)}`);

console.log('\n=== Test Complete ===');
console.log('Review the triples above. Each should have:');
console.log('  1. Correct actor identification (Jeffrey Epstein if jeeitunes/jee)');
console.log('  2. A tags array with 2-6 descriptive tags');
console.log('  3. An explicit_topic (what it directly says)');
console.log('  4. An implicit_topic (what it likely implies)');
console.log('  5. Appropriate categorization and context');

db.close();
