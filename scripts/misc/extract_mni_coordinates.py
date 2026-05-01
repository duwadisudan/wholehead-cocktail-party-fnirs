"""
Extract MNI xyz triplets from the AtlasViewer optode CSV (one-time utility).

Reads a CSV whose 4th column stores a single "x y z" string, splits that
triplet into three standalone columns, and writes the result as a new CSV
(no header) next to the source file by default. Used during MNI/atlas
preparation prior to classifier training.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring, documentation, and commenting were AI-assisted;
       all scientific decisions and accountability remain with the author.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

# CHANGE THIS FOR EACH SUBJECT
SUBJECT_ID = "sub-663"

BASE_DIR = Path(r"U:\eng_research_hrc_binauralhearinglab\Sudan\Labs\Sen Lab\Research_projects\Whole_Head_Cocktail_party\Cocktail_party_whole_head_master_data")
DEFAULT_SOURCE = BASE_DIR / SUBJECT_ID / "nirs" / "atlasviewer_mni" / "atlasviewer_cortex_projected_mni_scanner.csv"


def parse_triplet(raw_value: str) -> List[str]:
    """Return the xyz components contained in a single string field."""
    cleaned = raw_value.strip().strip('"').translate(str.maketrans('', '', '[]()'))
    parts = [part for part in cleaned.replace(',', ' ').split() if part]
    if len(parts) != 3:
        raise ValueError(f"expected 3 components, found {len(parts)} in '{raw_value}'")
    return parts


def extract_xyz(row: Sequence[str], column_index: int) -> List[str]:
    """Return xyz triplet from the preferred column, falling back to adjacent columns."""
    try:
        field = row[column_index]
    except IndexError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"row has {len(row)} columns, expected index {column_index + 1}"
        ) from exc

    try:
        return parse_triplet(field)
    except ValueError:
        tail = row[column_index : column_index + 3]
        if len(tail) != 3:
            raise
        # If the file already stores x,y,z across individual columns, just clean them up.
        return [item.strip().strip('"') for item in tail]


def iter_triplets(
    source_rows: Iterable[Sequence[str]],
    column_index: int,
    skip_header: bool,
) -> Iterable[List[str]]:
    """Yield parsed xyz rows from the source CSV."""
    for row_number, row in enumerate(source_rows, start=1):
        if row_number == 1 and skip_header:
            continue
        try:
            yield extract_xyz(row, column_index)
        except ValueError as exc:
            raise ValueError(f"row {row_number}: {exc}") from exc


def write_triplets(rows: Iterable[Sequence[str]], destination: Path) -> int:
    """Write xyz rows to destination CSV and return row count."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'source',
        nargs='?',
        type=Path,
        default=DEFAULT_SOURCE,
        help='Path to atlasviewer CSV input file (defaults to lab master file)',
    )
    parser.add_argument(
        '-c',
        '--column',
        type=int,
        default=4,
        help='1-based column number that contains the "x y z" string (default: 4)',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=None,
        help='Optional destination CSV path (defaults to just_mni.csv next to source)',
    )
    parser.add_argument(
        '--skip-header',
        action='store_true',
        help='Skip the first row if it is a header',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    column_index = args.column - 1
    if column_index < 0:
        parser.error('column number must be >= 1')

    source: Path = args.source
    if not source.exists():
        parser.error(f'source file not found: {source}')

    output: Path = args.output or source.with_name('just_mni.csv')

    with source.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        try:
            written = write_triplets(
                iter_triplets(reader, column_index, args.skip_header),
                output,
            )
        except ValueError as err:
            parser.error(str(err))

    print(f'Wrote {written} MNI rows to {output}')
    return 0


if __name__ == '__main__':
    # Just run with hardcoded settings when executed directly
    source = DEFAULT_SOURCE
    column_index = 3  # 4th column (0-indexed)
    skip_header = False  # Source file has no header - all rows are data
    output = source.with_name('just_mni.csv')
    
    if not source.exists():
        print(f'Error: source file not found: {source}')
        sys.exit(1)
    
    with source.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        try:
            written = write_triplets(
                iter_triplets(reader, column_index, skip_header),
                output,
            )
        except ValueError as err:
            print(f'Error: {err}')
            sys.exit(1)
    
    print(f'Wrote {written} MNI rows to {output}')
    sys.exit(0)
