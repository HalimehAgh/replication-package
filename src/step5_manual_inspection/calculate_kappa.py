"""
Cohen's Kappa Calculator for Manual Coding Inter-Rater Reliability
===================================================================
Usage:
    python calculate_kappa.py --supervisor supervisor.csv --student student.csv

Computes Cohen's kappa for each coding column and overall.
Works for both binary (Yes/No) columns and nominal (confound_type) columns.
"""

import csv
import argparse
from collections import defaultdict


def read_csv(path):
    """Read coding CSV and index rows by (commit_hash_8, smell_type, dimension)."""
    rows = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('pure_smell_fix', '').strip():  # skip empty rows
                key = (
                    row['after_commit'][:8],
                    row['smell_type'].strip(),
                    row['dimension'].strip()
                )
                rows[key] = row
    return rows


def cohen_kappa_binary(col, coder1, coder2):
    """
    Compute Cohen's kappa for a binary (Yes/No) column.
    Returns: (percent_agreement, p_expected, kappa, n)
    """
    keys = [k for k in coder1 if k in coder2]
    n = len(keys)
    if n == 0:
        return 0, 0, 0, 0

    agree = 0
    yes1 = 0
    yes2 = 0

    for k in keys:
        v1 = coder1[k].get(col, '').strip().lower()
        v2 = coder2[k].get(col, '').strip().lower()
        if v1 == v2:
            agree += 1
        if v1 == 'yes':
            yes1 += 1
        if v2 == 'yes':
            yes2 += 1

    po = agree / n
    # Expected agreement by chance
    p_yes = (yes1 / n) * (yes2 / n)
    p_no  = ((n - yes1) / n) * ((n - yes2) / n)
    pe = p_yes + p_no

    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 1.0
    return po, pe, kappa, n


def cohen_kappa_nominal(col, coder1, coder2):
    """
    Compute Cohen's kappa for a nominal (multi-category) column.
    Returns: (percent_agreement, p_expected, kappa, n)
    """
    keys = [k for k in coder1 if k in coder2]
    n = len(keys)
    if n == 0:
        return 0, 0, 0, 0

    # Collect all categories
    categories = set()
    for k in keys:
        categories.add(coder1[k].get(col, '').strip().lower())
        categories.add(coder2[k].get(col, '').strip().lower())

    agree = sum(
        1 for k in keys
        if coder1[k].get(col, '').strip().lower() == coder2[k].get(col, '').strip().lower()
    )
    po = agree / n

    # Expected agreement by chance (sum of product of marginal proportions)
    pe = 0
    for cat in categories:
        p1 = sum(1 for k in keys if coder1[k].get(col, '').strip().lower() == cat) / n
        p2 = sum(1 for k in keys if coder2[k].get(col, '').strip().lower() == cat) / n
        pe += p1 * p2

    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 1.0
    return po, pe, kappa, n


def interpret(kappa):
    """Landis & Koch (1977) kappa interpretation scale."""
    if kappa < 0:
        return "Poor (< 0)"
    elif kappa < 0.20:
        return "Slight (0.00-0.20)"
    elif kappa < 0.40:
        return "Fair (0.21-0.40)"
    elif kappa < 0.60:
        return "Moderate (0.41-0.60)"
    elif kappa < 0.80:
        return "Substantial (0.61-0.80)"
    else:
        return "Almost Perfect (0.81-1.00)"


def print_disagreements(coder1, coder2, cols):
    """Print all disagreements between coders."""
    print("\nDISAGREEMENTS:")
    print("=" * 110)
    total_disagree = 0
    for key in coder1:
        if key not in coder2:
            print(f"  MISSING in coder2: {key}")
            continue
        row_disagree = []
        for col in cols:
            v1 = coder1[key].get(col, '').strip()
            v2 = coder2[key].get(col, '').strip()
            if v1.lower() != v2.lower():
                row_disagree.append((col, v1, v2))
        if row_disagree:
            print(f"\n  Commit: {key[0]} | Smell: {key[1]} | Dim: {key[2]}")
            for col, v1, v2 in row_disagree:
                print(f"    {col:<25} Coder1: {v1:<20} Coder2: {v2}")
            total_disagree += len(row_disagree)
    print(f"\n  Total disagreements: {total_disagree}")


def export_disagreements_csv(coder1, coder2, cols, output_path):
    """Export all disagreements to a CSV file for discussion."""
    rows = []
    for key in coder1:
        if key not in coder2:
            continue
        for col in cols:
            v1 = coder1[key].get(col, '').strip()
            v2 = coder2[key].get(col, '').strip()
            if v1.lower() != v2.lower():
                rows.append({
                    'commit_hash':    key[0],
                    'smell_type':     key[1],
                    'dimension':      key[2],
                    'column':         col,
                    'supervisor_H':   v1,
                    'student_V':      v2,
                    'agreed_value':   '',   # to be filled in during meeting
                    'resolution_note': ''  # brief justification
                })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['commit_hash', 'smell_type', 'dimension', 'column',
                      'supervisor_H', 'student_V', 'agreed_value', 'resolution_note']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDisagreements CSV exported to: {output_path}")
    print(f"Total rows: {len(rows)}")
    print("Fill in 'agreed_value' and 'resolution_note' columns during the meeting.")


def main():
    parser = argparse.ArgumentParser(description="Compute Cohen's Kappa for manual coding CSVs")
    parser.add_argument('--supervisor', required=True, help='Path to supervisor CSV')
    parser.add_argument('--student',    required=True, help='Path to student CSV')
    parser.add_argument('--show-disagreements', action='store_true',
                        help='Print all disagreements')
    parser.add_argument('--export-disagreements', default='disagreements.csv',
                        help='Path for disagreements CSV output (default: disagreements.csv)')
    args = parser.parse_args()

    print(f"\nReading supervisor CSV: {args.supervisor}")
    print(f"Reading student CSV:    {args.student}")

    coder1 = read_csv(args.supervisor)
    coder2 = read_csv(args.student)

    print(f"\nRows in supervisor: {len(coder1)}")
    print(f"Rows in student:    {len(coder2)}")
    print(f"Common rows:        {len([k for k in coder1 if k in coder2])}")

    binary_cols = [
        'pure_smell_fix',
        'method_extraction',
        'error_handling',
        'type_hints',
        'documentation',
        'refactored_nearby',
        'defensive_checks',
    ]
    nominal_cols = ['confound_type']
    all_cols = binary_cols + nominal_cols

    print("\n" + "=" * 80)
    print("COHEN'S KAPPA PER COLUMN")
    print("=" * 80)
    print(f"{'Column':<28} {'Agree%':>8} {'Po':>6} {'Pe':>6} {'Kappa':>8}  {'Interpretation'}")
    print("-" * 80)

    all_kappas = []

    for col in binary_cols:
        po, pe, kappa, n = cohen_kappa_binary(col, coder1, coder2)
        all_kappas.append(kappa)
        print(f"{col:<28} {po*100:>7.1f}% {po:>6.3f} {pe:>6.3f} {kappa:>8.3f}  {interpret(kappa)}")

    for col in nominal_cols:
        po, pe, kappa, n = cohen_kappa_nominal(col, coder1, coder2)
        all_kappas.append(kappa)
        print(f"{col:<28} {po*100:>7.1f}% {po:>6.3f} {pe:>6.3f} {kappa:>8.3f}  {interpret(kappa)}")

    print("-" * 80)
    overall = sum(all_kappas) / len(all_kappas)
    total_comparisons = sum(
        len([k for k in coder1 if k in coder2]) for _ in all_cols
    )
    total_agree = sum(
        sum(1 for k in coder1 if k in coder2
            and coder1[k].get(col,'').strip().lower() == coder2[k].get(col,'').strip().lower())
        for col in all_cols
    )
    print(f"{'OVERALL (mean kappa)':<28} {total_agree/total_comparisons*100:>7.1f}% "
          f"{'':>6} {'':>6} {overall:>8.3f}  {interpret(overall)}")
    print(f"\nTotal comparisons: {total_comparisons}")
    print(f"Total agreements:  {total_agree}")
    print(f"Total disagreements: {total_comparisons - total_agree}")

    print("\n" + "=" * 80)
    print("INTERPRETATION SCALE (Landis & Koch, 1977)")
    print("=" * 80)
    bands = [
        (float('-inf'), 0.00, "Poor          (< 0.00)"),
        (0.00,          0.20, "Slight        (0.00-0.20)"),
        (0.20,          0.40, "Fair          (0.21-0.40)"),
        (0.40,          0.60, "Moderate      (0.41-0.60)"),
        (0.60,          0.80, "Substantial   (0.61-0.80)"),
        (0.80,          1.01, "Almost Perfect(0.81-1.00)"),
    ]
    for lo, hi, label in bands:
        marker = "  ← your result" if lo <= overall < hi else ""
        print(f"  {label}{marker}")

    if args.show_disagreements:
        print_disagreements(coder1, coder2, all_cols)

    # Always export disagreements CSV
    export_disagreements_csv(coder1, coder2, all_cols, args.export_disagreements)


if __name__ == '__main__':
    main()