#!/usr/bin/env python3
"""
Extract text data from DOJ document collection files.
Parses .dat and .opt files to extract document metadata.
"""

import csv
import sys
from pathlib import Path


def parse_dat_file(dat_path):
    """
    Parse the .dat file which contains document IDs and PDF filenames.
    Uses 0xFE as delimiter.
    """
    print(f"\n{'='*70}")
    print(f"Parsing DAT file: {dat_path}")
    print(f"{'='*70}\n")

    with open(dat_path, 'rb') as f:
        content = f.read()

    # Split by 0xFE delimiter
    parts = content.split(b'\xfe')

    documents = []
    current_doc = {}

    # Parse the fields
    for i, part in enumerate(parts):
        decoded = part.decode('utf-8', errors='ignore').strip()
        if not decoded or decoded in ['Prod Beg', 'Prod End', 'Filename', 'FILE_PATH']:
            continue

        # Look for document ID pattern
        if decoded.startswith('DOJ-OGR-'):
            if 'doc_id_start' not in current_doc:
                current_doc['doc_id_start'] = decoded
            elif 'doc_id_end' not in current_doc:
                current_doc['doc_id_end'] = decoded
        # Look for PDF filename
        elif '.pdf' in decoded.lower():
            current_doc['filename'] = decoded
            # Save document when we have all parts
            if 'doc_id_start' in current_doc and 'doc_id_end' in current_doc:
                documents.append(current_doc.copy())
                current_doc = {}

    # Print summary
    print(f"Total documents found: {len(documents)}\n")
    print("Sample documents:")
    print("-" * 70)
    for i, doc in enumerate(documents[:10]):
        print(f"{i+1}. {doc.get('doc_id_start', 'N/A')} - {doc.get('doc_id_end', 'N/A')}")
        print(f"   File: {doc.get('filename', 'N/A')}")
        print()

    if len(documents) > 10:
        print(f"... and {len(documents) - 10} more documents\n")

    return documents


def parse_opt_file(opt_path):
    """
    Parse the .opt CSV file which contains image references.
    """
    print(f"\n{'='*70}")
    print(f"Parsing OPT file: {opt_path}")
    print(f"{'='*70}\n")

    images = []

    with open(opt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                images.append({
                    'doc_id': parts[0],
                    'volume': parts[1],
                    'image_path': parts[2],
                    'flag': parts[3] if len(parts) > 3 else '',
                })

    print(f"Total image references: {len(images)}\n")
    print("Sample image references:")
    print("-" * 70)
    for i, img in enumerate(images[:10]):
        print(f"{i+1}. {img['doc_id']}: {img['image_path']}")

    if len(images) > 10:
        print(f"... and {len(images) - 10} more images\n")

    return images


def export_to_text(documents, images, output_dir):
    """
    Export extracted data to text files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Export documents list
    doc_file = output_path / "documents_list.txt"
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write("DOJ Document Collection - Extracted Documents\n")
        f.write("=" * 70 + "\n\n")
        for i, doc in enumerate(documents, 1):
            f.write(f"{i}. Document ID Range: {doc.get('doc_id_start')} - {doc.get('doc_id_end')}\n")
            f.write(f"   Filename: {doc.get('filename')}\n\n")

    print(f"✓ Exported documents list to: {doc_file}")

    # Export images list
    img_file = output_path / "images_list.txt"
    with open(img_file, 'w', encoding='utf-8') as f:
        f.write("DOJ Document Collection - Image References\n")
        f.write("=" * 70 + "\n\n")
        for i, img in enumerate(images, 1):
            f.write(f"{i}. {img['doc_id']}: {img['image_path']}\n")

    print(f"✓ Exported images list to: {img_file}")

    # Export CSV versions
    doc_csv = output_path / "documents.csv"
    with open(doc_csv, 'w', newline='', encoding='utf-8') as f:
        if documents:
            writer = csv.DictWriter(f, fieldnames=documents[0].keys())
            writer.writeheader()
            writer.writerows(documents)

    print(f"✓ Exported documents CSV to: {doc_csv}")

    img_csv = output_path / "images.csv"
    with open(img_csv, 'w', newline='', encoding='utf-8') as f:
        if images:
            writer = csv.DictWriter(f, fieldnames=images[0].keys())
            writer.writeheader()
            writer.writerows(images)

    print(f"✓ Exported images CSV to: {img_csv}")


def main():
    data_dir = Path("data")

    # Find .dat and .opt files
    dat_files = list(data_dir.glob("*.dat"))
    opt_files = list(data_dir.glob("*.opt"))

    if not dat_files and not opt_files:
        print("Error: No .dat or .opt files found in data directory")
        sys.exit(1)

    all_documents = []
    all_images = []

    # Parse all .dat files
    for dat_file in dat_files:
        docs = parse_dat_file(dat_file)
        all_documents.extend(docs)

    # Parse all .opt files
    for opt_file in opt_files:
        imgs = parse_opt_file(opt_file)
        all_images.extend(imgs)

    # Export extracted data
    if all_documents or all_images:
        print(f"\n{'='*70}")
        print("Exporting extracted data...")
        print(f"{'='*70}\n")
        export_to_text(all_documents, all_images, "extracted")

        print(f"\n{'='*70}")
        print("Extraction Complete!")
        print(f"{'='*70}")
        print(f"Total documents extracted: {len(all_documents)}")
        print(f"Total images extracted: {len(all_images)}")
        print("\nExtracted files saved to 'extracted/' directory")


if __name__ == "__main__":
    main()
