"""IDX Exchange | Week 5: finalize MLS data cleaning.

Run from the project directory after week4_data_cleaning.py.
All original rows are retained; questionable data is flagged rather than deleted.
Outputs contain confidential MLS records: do not commit CSV files to GitHub.
"""
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
# Approximate California bounding box for initial screening, not a state-border test.
CA_LAT_MIN, CA_LAT_MAX = 32.0, 42.1
CA_LON_MIN, CA_LON_MAX = -124.5, -114.0

CORE_FIELDS = {
    'ListingKey', 'PropertyType', 'PropertySubType', 'CloseDate',
    'PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate',
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
    'DaysOnMarket', 'BedroomsTotal', 'BathroomsTotalInteger',
    'Latitude', 'Longitude', 'CountyOrParish', 'City', 'PostalCode',
    'MLSAreaMajor', 'ListOfficeName', 'BuyerOfficeName', 'year_month',
    'rate_30yr_fixed',
}
DATE_COLS = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate',
             'ContractStatusChangeDate']
NUMERIC_COLS = ['ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
                'DaysOnMarket', 'BedroomsTotal', 'BathroomsTotalInteger',
                'Latitude', 'Longitude']


def clean_dataset(name):
    path = BASE / f'week4_{name}_flagged.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Run Week 4 first or move its output here.')
    df = pd.read_csv(path, low_memory=False)
    before_rows, before_cols = df.shape
    print(f'\n{name.upper()} BEFORE: {before_rows:,} rows, {before_cols:,} columns')

    for col in DATE_COLS:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    for col in NUMERIC_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Missing values remain NaN (unknown), not zero. Document rather than
    # deleting incomplete records or filling market prices with invented values.
    missing_before = df.isna().mean().mul(100)
    high_missing = missing_before[missing_before > 90].sort_values(ascending=False)
    high_missing.rename('missing_pct').to_csv(BASE / f'week5_{name}_high_missing_review.csv')
    print('Columns >90% missing (review only; no automatic deletion):', len(high_missing))

    # Date comparisons require both dates; missing dates do not imply an error.
    listing = df['ListingContractDate'] if 'ListingContractDate' in df else pd.Series(pd.NaT, index=df.index)
    purchase = df['PurchaseContractDate'] if 'PurchaseContractDate' in df else pd.Series(pd.NaT, index=df.index)
    close = df['CloseDate'] if 'CloseDate' in df else pd.Series(pd.NaT, index=df.index)
    df['listing_after_close_flag'] = (listing > close).fillna(False)
    df['purchase_after_close_flag'] = (purchase > close).fillna(False)
    df['negative_timeline_flag'] = ((listing > purchase) | (listing > close) |
                                    (purchase > close)).fillna(False)

    lat = df['Latitude'] if 'Latitude' in df else pd.Series(np.nan, index=df.index)
    lon = df['Longitude'] if 'Longitude' in df else pd.Series(np.nan, index=df.index)
    df['missing_coordinates_flag'] = lat.isna() | lon.isna()
    df['zero_coordinates_flag'] = (lat == 0) | (lon == 0)
    df['positive_longitude_flag'] = lon > 0
    df['implausible_coordinates_flag'] = (
        lat.notna() & lon.notna() &
        (~lat.between(CA_LAT_MIN, CA_LAT_MAX) | ~lon.between(CA_LON_MIN, CA_LON_MAX))
    )
    geo_flags = ['missing_coordinates_flag', 'zero_coordinates_flag',
                 'positive_longitude_flag', 'implausible_coordinates_flag']
    date_flags = ['listing_after_close_flag', 'purchase_after_close_flag',
                  'negative_timeline_flag']

    # Preserve the Week 4 invalid-value flags, but convert impossible numeric
    # entries to NaN for safer calculations; retain rows for later review.
    invalid_rules = {'ClosePrice': lambda x: x <= 0,
                     'LivingArea': lambda x: x <= 0,
                     'DaysOnMarket': lambda x: x < 0,
                     'BedroomsTotal': lambda x: x < 0,
                     'BathroomsTotalInteger': lambda x: x < 0}
    for col, rule in invalid_rules.items():
        if col in df:
            bad = rule(df[col])
            df.loc[bad, col] = np.nan

    # Avoid speculative removal of metadata/redundant fields: review candidates.
    # Only drop truly empty non-core columns; all core fields are preserved.
    empty_noncore = [c for c in df if c not in CORE_FIELDS and df[c].isna().all()]
    if empty_noncore:
        df = df.drop(columns=empty_noncore)
    pd.Series(empty_noncore, name='dropped_fully_empty_noncore_columns').to_csv(
        BASE / f'week5_{name}_dropped_columns.csv', index=False)

    summary = {**{c: int(df[c].sum()) for c in date_flags + geo_flags},
               'any_geographic_issue': int(df[geo_flags].any(axis=1).sum()),
               'rows_before': before_rows, 'rows_after': len(df),
               'columns_before': before_cols, 'columns_after': df.shape[1],
               'fully_empty_noncore_columns_dropped': len(empty_noncore)}
    pd.Series(summary, name='count').to_csv(BASE / f'week5_{name}_quality_summary.csv')
    df.dtypes.astype(str).rename('dtype').to_csv(BASE / f'week5_{name}_dtypes.csv')
    output = BASE / f'week5_{name}_cleaned.csv'
    df.to_csv(output, index=False)
    print(pd.Series(summary).to_string())
    print('Date column types:', df[[c for c in DATE_COLS if c in df]].dtypes.to_dict())
    print('Numeric column types:', df[[c for c in NUMERIC_COLS if c in df]].dtypes.to_dict())
    print('Saved:', output)


if __name__ == '__main__':
    for dataset in ['sold', 'listings']:
        clean_dataset(dataset)

# I removed columns that contained entirely missing values and were not essential to the analysis.
# These columns would not contribute meaningful information to future calculations or visualizations. 
# However, I retained core fields such as property prices, dates, locations, and property characteristics, 
# even when they contained missing values, because they are important for future market analysis.