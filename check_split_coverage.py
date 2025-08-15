#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_split_coverage.py

Orthogonal sanity checks for scaffold-based dataset splits in multitask (protein-column) matrices.

What it does:
1) Confirms no duplicate SMILES across splits.
2) Optional: computes Bemis–Murcko scaffolds (if RDKit is installed) and checks that
   no scaffold appears in more than one split (leakage).
3) Computes per-protein coverage (non-NaN counts) for train/val/test and flags targets
   that have zero samples in any split or fail a minimum-per-split threshold.
4) Writes a CSV summary and prints a concise text report.

Usage example:
python check_split_coverage.py --train chembl_pretraining_train.csv --val chembl_pretraining_val.csv --test chembl_pretraining_test.csv --min-per-split 1 --out report_split_check.csv
"""
import argparse
import sys
import os
import pandas as pd

def try_import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        return Chem, MurckoScaffold
    except Exception:
        return None, None

def bm_scaffold(smiles, Chem, MurckoScaffold):
    if Chem is None:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    sc = MurckoScaffold.GetScaffoldForMol(m)
    if sc is None or sc.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(sc, isomericSmiles=False)

def load_matrix(path):
    df = pd.read_csv(path)
    if 'smiles' not in df.columns:
        raise ValueError(f"{path}: expected a 'smiles' column")
    # Keep columns ordered: smiles + protein columns
    prot_cols = [c for c in df.columns if c != 'smiles']
    return df[['smiles'] + prot_cols]

def compute_scaffolds(df, Chem, MurckoScaffold):
    if Chem is None:
        return pd.Series([None] * len(df), index=df.index)
    return df['smiles'].map(lambda s: bm_scaffold(s, Chem, MurckoScaffold))

def summarize_coverage(dfs, names):
    # dfs is dict of name->df; names is list in order like ['train','val','test']
    protein_cols = [c for c in dfs[names[0]].columns if c != 'smiles']
    # Confirm columns match across splits
    for nm in names[1:]:
        other = [c for c in dfs[nm].columns if c != 'smiles']
        if protein_cols != other:
            raise ValueError("Protein columns differ across splits.")
    # Build summary frame: rows=protein, columns=count_{split}
    summary = []
    for p in protein_cols:
        row = {'protein': p}
        for nm in names:
            row[f'count_{nm}'] = int(dfs[nm][p].notna().sum())
        summary.append(row)
    cov = pd.DataFrame(summary).set_index('protein')
    cov['count_total'] = cov.sum(axis=1)
    return cov

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True, help='Path to train CSV')
    ap.add_argument('--val', required=True, help='Path to val CSV')
    ap.add_argument('--test', required=True, help='Path to test CSV')
    ap.add_argument('--min-per-split', type=int, default=1, help='Minimum non-NaN per protein per split (default: 1)')
    ap.add_argument('--out', default='split_orthogonal_report.csv', help='Output CSV path')
    args = ap.parse_args()

    # Load
    dfs = {
        'train': load_matrix(args.train),
        'val':   load_matrix(args.val),
        'test':  load_matrix(args.test),
    }
    names = ['train','val','test']

    # 1) Duplicates across splits
    all_smiles = []
    for nm in names:
        all_smiles.append(pd.Series(dfs[nm]['smiles'], name=nm))
    sm = pd.concat(
        [dfs[nm][['smiles']].assign(which=nm) for nm in names],
        ignore_index=True
    )
    dup_smiles = sm.groupby('smiles', dropna=False)['which'].nunique()
    dup_violations = dup_smiles[dup_smiles > 1].index.tolist()

    # 2) Scaffold leakage (optional RDKit)
    Chem, MurckoScaffold = try_import_rdkit()
    scaf_map = {}
    scaf_leak = set()
    if Chem is not None:
        for nm in names:
            scafs = compute_scaffolds(dfs[nm], Chem, MurckoScaffold)
            for s in scafs:
                if s is None:
                    continue
                if s in scaf_map and scaf_map[s] != nm:
                    scaf_leak.add(s)
                else:
                    scaf_map[s] = nm

    # 3) Per-protein coverage
    cov = summarize_coverage(dfs, names)
    # Flag proteins
    zero_any = cov[[f'count_{nm}' for nm in names]].min(axis=1) == 0
    below_min = (cov[[f'count_{nm}' for nm in names]] < args.min_per_split).any(axis=1)
    cov['flag_zero_in_a_split'] = zero_any
    cov['flag_below_min_in_a_split'] = below_min

    # 4) Save report
    cov.to_csv(args.out, index=True)

    # Print concise report
    print("=== Orthogonal Split Check ===")
    print(f"Total proteins: {cov.shape[0]}")
    print(f"Proteins with ZERO samples in at least one split: {int(zero_any.sum())}")
    print(f"Proteins below min-per-split ({args.min_per_split}) in at least one split: {int(below_min.sum())}")
    top_missing = cov[zero_any].sort_values('count_total').head(20)
    if not top_missing.empty:
        print("\nExamples (up to 20) of proteins missing in a split:")
        for prot, row in top_missing.iterrows():
            print(f"  {prot}: train={row['count_train']}, val={row['count_val']}, test={row['count_test']} (total={row['count_total']})")
    # Duplicates
    if dup_violations:
        print(f"\n[FAIL] Duplicate SMILES found across splits: {len(dup_violations)}")
    else:
        print("\n[OK] No duplicate SMILES across splits.")
    # Scaffold leakage
    if Chem is None:
        print("[WARN] RDKit not available; scaffold leakage check skipped.")
    else:
        if scaf_leak:
            print(f"[FAIL] Shared Bemis–Murcko scaffolds across splits: {len(scaf_leak)}")
        else:
            print("[OK] No Bemis–Murcko scaffold leakage across splits.")
    print(f"\nDetailed per-protein coverage saved to: {os.path.abspath(args.out)}")

if __name__ == '__main__':
    main()
