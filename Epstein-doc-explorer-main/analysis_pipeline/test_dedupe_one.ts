import { query } from '@anthropic-ai/claude-agent-sdk';

async function testOneGroup() {
  const testNames = ['Jeffrey Epstein', 'Jeffrey E.', 'Jeff Epstein', 'J. Epstein'];

  const prompt = `Analyze these entity names and determine which should be merged (same person) vs kept separate (different people).

RULES:
- DO merge: name variations, nicknames, case differences (Jeffrey Epstein = Jeff Epstein = jeffrey epstein)
- DO NOT merge: numbered entities (Jane Doe 1 ≠ Jane Doe 2), family members (George H.W. Bush ≠ George W. Bush), generic vs specific (Jeffrey ≠ Jeffrey Epstein)

Names (${testNames.length} total):
${testNames.map((n, i) => `${i + 1}. ${n}`).join('\n')}

Return ONLY valid JSON with this exact structure:
{
  "merge_groups": [
    {"canonical": "Best Full Name", "aliases": ["variant1", "variant2"], "reasoning": "why same person"}
  ],
  "do_not_merge": ["name1"],
  "reasoning_for_no_merge": "why separate"
}

If no merges needed, use empty array: {"merge_groups": [], "do_not_merge": ${JSON.stringify(testNames)}, "reasoning_for_no_merge": "all distinct"}`;

  console.log('=== Testing Single Group ===\n');
  console.log('Names:', testNames);
  console.log('\nSending prompt to Claude...\n');

  let responseText = '';

  const agent = query({
    prompt,
    options: {
      model: 'claude-haiku-4-5',
      maxTokens: 4096,
      maxTurns: 1,
      allowedTools: [],
    }
  });

  try {
    console.log('Collecting response...\n');
    for await (const message of agent) {
      console.log('Message type:', message.type);
      if (message.type === 'assistant') {
        const textBlocks = message.message.content.filter((c: any) => c.type === 'text');
        for (const block of textBlocks) {
          responseText += block.text;
        }
      }
    }

    console.log('\n=== Response Text ===');
    console.log(responseText);
    console.log('\n=== Attempting JSON Extraction ===');

    if (!responseText) {
      console.log('ERROR: No response text found!');
      return;
    }

    // Try to extract JSON
    let jsonText = responseText;
    const codeBlockMatch = responseText.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (codeBlockMatch) {
      console.log('Found JSON in code block');
      jsonText = codeBlockMatch[1];
    } else {
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        console.log('Found JSON in raw text');
        jsonText = jsonMatch[0];
      } else {
        console.log('ERROR: No JSON found in response');
        return;
      }
    }

    console.log('\nExtracted JSON:');
    console.log(jsonText);

    const parsed = JSON.parse(jsonText.trim());
    console.log('\n=== Parsed Result ===');
    console.log(JSON.stringify(parsed, null, 2));

    console.log('\n✓ Success! Merge groups:', parsed.merge_groups?.length || 0);
  } catch (error) {
    console.error('\n✗ Error:', error);
  }
}

testOneGroup().catch(console.error);
