"""IDX Exchange | Week 7: Outlier Detection and Data Quality.

Run after Week 6. Keeps all records in a flagged export and writes a
separate filtered analysis dataset. No original file is overwritten.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT = BASE / 'week6_sold_features.csv'
FIELDS = ['ClosePrice', 'LivingArea', 'DaysOnMarket']


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f'{INPUT} not found. Run Week 6 first.')

    sold = pd.read_csv(INPUT, low_memory=False)
    missing = [c for c in FIELDS if c not in sold.columns]
    if missing:
        raise KeyError(f'Missing required columns: {missing}')
    print('Original dataset shape:', sold.shape)

    for col in FIELDS:
        sold[col] = pd.to_numeric(sold[col], errors='coerce')
        q1, q3 = sold[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        # Missing values are tracked separately, not mislabeled as IQR outliers.
        sold[f'{col}_outlier_flag'] = sold[col].notna() & ~sold[col].between(lower, upper)
        sold[f'{col}_missing_flag'] = sold[col].isna()
        print(f'{col}: Q1={q1:,.2f}, Q3={q3:,.2f}, IQR={iqr:,.2f}, '
              f'bounds=({lower:,.2f}, {upper:,.2f}), '
              f'outliers={sold[f"{col}_outlier_flag"].sum():,}, '
              f'missing={sold[f"{col}_missing_flag"].sum():,}')

    outlier_cols = [f'{c}_outlier_flag' for c in FIELDS]
    sold['any_outlier_flag'] = sold[outlier_cols].any(axis=1)
    sold['invalid_numeric_flag'] = (
        (sold['ClosePrice'] <= 0) |
        (sold['LivingArea'] <= 0) |
        (sold['DaysOnMarket'] < 0)
    )
    # Keep records with missing values in the flagged export; exclude them
    # from the filtered analysis dataset for these three core measures.
    sold['missing_core_metric_flag'] = sold[FIELDS].isna().any(axis=1)
    excluded = (sold['any_outlier_flag'] |
                sold['invalid_numeric_flag'] |
                sold['missing_core_metric_flag'])

    sold['excluded_from_analysis'] = excluded

    filtered = sold.loc[~excluded].copy()

    comparison = pd.DataFrame({
        'metric': FIELDS,
        'median_before': [sold[c].median() for c in FIELDS],
        'median_after': [filtered[c].median() for c in FIELDS],
    })
    comparison['median_change'] = comparison['median_after'] - comparison['median_before']

    print('\nDataset size comparison:')
    print(f'Before filtering: {len(sold):,} rows')
    print(f'After filtering:  {len(filtered):,} rows')
    print(f'Excluded:         {excluded.sum():,} rows ({excluded.mean():.2%})')
    print(f'Records retained: {len(filtered) / len(sold):.2%}' if len(sold) else 'Records retained: N/A (empty dataset)')
    print(f'IQR outliers (any): {sold["any_outlier_flag"].sum():,}')
    print(f'Invalid numeric values (any): {sold["invalid_numeric_flag"].sum():,}')
    print(f'Missing core metrics (any): {sold["missing_core_metric_flag"].sum():,}')
    print('\nMedian comparison:')
    print(comparison.to_string(index=False))

    sold.to_csv(BASE / 'week7_sold_flagged.csv', index=False)
    filtered.to_csv(BASE / 'week7_sold_filtered.csv', index=False)
    comparison.to_csv(BASE / 'week7_median_comparison.csv', index=False)
    with (BASE / 'week7_comparison.txt').open('w') as f:
        f.write('Week 7: Outlier Detection and Data Quality\n')
        f.write(f'Rows before: {len(sold):,}\nRows after: {len(filtered):,}\n')
        f.write(f'Rows excluded: {excluded.sum():,} ({excluded.mean():.2%})\n')
        f.write('IQR flags are calculated separately for ClosePrice, LivingArea, '
                'and DaysOnMarket using 1.5 x IQR bounds.\n')
        f.write('The filtered dataset excludes any IQR outlier, invalid numeric '
                'value, or missing core metric; the flagged dataset preserves all rows.\n\n')
        f.write(comparison.to_string(index=False))
        f.write('\n')
    print('\nSaved flagged and filtered datasets, median comparison CSV, and written report.')


if __name__ == '__main__':
    main()



