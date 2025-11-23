#!/bin/bash

while true; do
  clear
  echo "=== Document Analysis Progress ==="
  echo ""

  if [ -f "document_analysis.db" ]; then
    docs=$(sqlite3 document_analysis.db "SELECT COUNT(*) FROM documents;" 2>/dev/null || echo "0")
    triples=$(sqlite3 document_analysis.db "SELECT COUNT(*) FROM rdf_triples;" 2>/dev/null || echo "0")
    errors=$(sqlite3 document_analysis.db "SELECT COUNT(*) FROM documents WHERE error IS NOT NULL;" 2>/dev/null || echo "0")

    progress=$(echo "scale=1; $docs * 100 / 2307" | bc)

    echo "Documents analyzed: $docs / 2307 ($progress%)"
    echo "RDF triples extracted: $triples"
    echo "Errors: $errors"
    echo ""
    echo "Last 5 analyzed:"
    sqlite3 document_analysis.db -column "SELECT doc_id, category FROM documents ORDER BY created_at DESC LIMIT 5;" 2>/dev/null
  else
    echo "Database not yet created..."
  fi

  sleep 30
done
