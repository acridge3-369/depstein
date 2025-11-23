#!/usr/bin/env node

import { query } from '@anthropic-ai/claude-agent-sdk';

const ANALYSIS_MODEL = 'claude-haiku-4-5';

// Intentionally broken JSON - missing quote, trailing comma, unescaped quote
const brokenJson = `{
  "one_sentence_summary": "A test document about Jeffrey Epstein's activities",
  "paragraph_summary": "This document discusses various activities and it contains an unescaped " quote here.",
  "date_range_earliest": "2005-01-01",
  "date_range_latest": "2010-12-31",
  "category": "email,
  "content_tags": ["legal_proceedings", "financial_transactions"],
  "rdf_triples": [
    {
      "timestamp": "2005-06-15",
      "actor": "Jeffrey Epstein",
      "action": "met with",
      "target": "Donald Trump",
      "tags": ["social_event", "high_profile_meeting"],
      "explicit_topic": "social meeting",
      "implicit_topic": "relationship building"
    }
  ]
}`;

console.log('=== Testing JSON Repair Mechanism ===\n');
console.log('Broken JSON:');
console.log(brokenJson);
console.log('\n');

// Try to parse - this should fail
let parsed;
try {
  parsed = JSON.parse(brokenJson);
  console.log('❌ Unexpected: JSON parsed successfully (it should have failed)');
  process.exit(1);
} catch (parseError) {
  console.log('✓ JSON parsing failed as expected');
  console.log(`Error: ${parseError instanceof Error ? parseError.message : String(parseError)}\n`);

  // Now attempt repair
  console.log('Attempting to repair...\n');

  const repairPrompt = `The following JSON response from a document analysis has a parsing error. Please identify and fix the issue, then return ONLY the corrected JSON:

\`\`\`json
${brokenJson}
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
      maxTokens: 4000,
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
    console.error('Response:', repairedText);
    process.exit(1);
  }

  const repairedJsonText = repairMatch[1] || repairMatch[0];

  console.log('Repaired JSON:');
  console.log(repairedJsonText);
  console.log('\n');

  try {
    parsed = JSON.parse(repairedJsonText);
    console.log('✓ JSON successfully repaired and parsed!\n');
    console.log('Parsed object:');
    console.log(JSON.stringify(parsed, null, 2));
    console.log('\n=== Repair Test PASSED ===');
  } catch (repairError) {
    console.error('❌ Repair failed: Still cannot parse JSON');
    console.error('Original error:', parseError);
    console.error('Repair error:', repairError);
    console.error('\nRepaired JSON:\n', repairedJsonText);
    process.exit(1);
  }
}
