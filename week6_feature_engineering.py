"""IDX Exchange | Week 6: feature engineering and market metrics"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SCHOOL_DISTRICTS_FILE = BASE / 'school_districts' / 'DistrictAreas2425' / 'DistrictAreas2425.shp'


def add_school_districts(df):
    """Optional point-in-polygon lookup; preserve rows and unknown districts."""
    df['school_district'] = pd.NA
    if SCHOOL_DISTRICTS_FILE is None:
        print('School districts: skipped (set SCHOOL_DISTRICTS_FILE to boundary data).')
        return df

    import geopandas as gpd
    districts = gpd.read_file(SCHOOL_DISTRICTS_FILE)
    if districts.crs is None:
        raise ValueError('School district boundaries have no CRS; verify source data.')
    print('School boundary fields:', districts.columns.tolist())
    # Adjust after inspecting the downloaded boundary file's actual schema.
    name_candidates = ['DistrictNa', 'NAME', 'Name', 'name', 'DISTRICT', 'District', 'district']
    name_col = next((c for c in name_candidates if c in districts.columns), None)
    if name_col is None:
        raise ValueError('Specify the school district name field based on printed boundary fields.')

    lat = pd.to_numeric(df['Latitude'], errors='coerce')
    lon = pd.to_numeric(df['Longitude'], errors='coerce')
    valid = lat.between(32.0, 42.1) & lon.between(-124.5, -114.0)
    if not valid.any():
        return df
    pts = gpd.GeoDataFrame(
        {'source_row': df.index[valid]},
        geometry=gpd.points_from_xy(lon[valid], lat[valid]),
        crs='EPSG:4326'
    ).to_crs(districts.crs)
    matches = gpd.sjoin(pts, districts[[name_col, 'geometry']], how='left', predicate='within')
    # District boundaries may overlap (e.g. elementary and unified districts).
    # Retain all distinct matches rather than duplicating property records.
    names = matches.groupby('source_row')[name_col].agg(
        lambda s: '; '.join(sorted(set(s.dropna().astype(str)))) or pd.NA
    )
    df.loc[names.index, 'school_district'] = names
    print('School district rows matched:', df['school_district'].notna().sum())
    return df


def main():
    input_path = BASE / 'week5_sold_cleaned.csv'
    if not input_path.exists():
        raise FileNotFoundError(f'{input_path} not found. Run Week 5 first.')

    sold = pd.read_csv(input_path, low_memory=False)
    original_rows = len(sold)
    print('Week 5 sold dataset shape:', sold.shape)

    dates = ['CloseDate', 'ListingContractDate', 'PurchaseContractDate']
    nums = ['ClosePrice', 'OriginalListPrice', 'LivingArea', 'DaysOnMarket']
    required = dates + nums + ['CountyOrParish', 'Latitude', 'Longitude']
    missing = [col for col in required if col not in sold.columns]
    if missing:
        raise KeyError(f'Missing required columns: {missing}')

    for col in dates:
        sold[col] = pd.to_datetime(sold[col], errors='coerce')
    for col in nums:
        sold[col] = pd.to_numeric(sold[col], errors='coerce')

    # Only positive denominators; missing/invalid values produce NaN, not infinity.
    sold['price_ratio'] = sold['ClosePrice'] / sold['OriginalListPrice'].where(sold['OriginalListPrice'] > 0)
    sold['close_to_original_list_ratio'] = sold['price_ratio']
    sold['price_per_sqft'] = sold['ClosePrice'] / sold['LivingArea'].where(sold['LivingArea'] > 0)
    sold['days_on_market'] = sold['DaysOnMarket']

    sold['year'] = sold['CloseDate'].dt.year.astype('Int64')
    sold['month'] = sold['CloseDate'].dt.month.astype('Int64')
    sold['YrMo'] = sold['CloseDate'].dt.strftime('%Y-%m')
    sold['listing_to_contract_days'] = (
        sold['PurchaseContractDate'] - sold['ListingContractDate']
    ).dt.days
    sold['contract_to_close_days'] = (
        sold['CloseDate'] - sold['PurchaseContractDate']
    ).dt.days

    # Do not silently change invalid timelines: flag for review.
    sold['invalid_listing_to_contract_flag'] = sold['listing_to_contract_days'] < 0
    sold['invalid_contract_to_close_flag'] = sold['contract_to_close_days'] < 0

    sold = add_school_districts(sold)

    features = ['price_ratio', 'close_to_original_list_ratio', 'price_per_sqft',
                'days_on_market', 'year', 'month', 'YrMo',
                'listing_to_contract_days', 'contract_to_close_days', 'school_district']
    print('\nSample engineered features:')
    print(sold[features].head(10).to_string(index=False))
    sold[features].head(10).to_csv(BASE / 'week6_sample_features.csv', index=False)

    # Segment summary uses valid values for each measure; no records removed.
    county_summary = (
        sold.groupby('CountyOrParish', dropna=False)
        .agg(total_sales=('ClosePrice', 'count'),
             median_close_price=('ClosePrice', 'median'),
             average_price_per_sqft=('price_per_sqft', 'mean'),
             average_days_on_market=('days_on_market', 'mean'),
             average_price_ratio=('price_ratio', 'mean'))
        .reset_index()
        .sort_values('total_sales', ascending=False)
    )
    print('\nCounty market summary (top 10 by sales):')
    print(county_summary.head(10).to_string(index=False))
    print('\nNegative listing-to-contract intervals:', sold['invalid_listing_to_contract_flag'].sum())
    print('Negative contract-to-close intervals:', sold['invalid_contract_to_close_flag'].sum())
    print('Rows retained:', len(sold), 'of', original_rows)

    sold.to_csv(BASE / 'week6_sold_features.csv', index=False)
    county_summary.to_csv(BASE / 'week6_county_summary.csv', index=False)
    print('\nSaved week6_sold_features.csv, week6_county_summary.csv, week6_sample_features.csv')


if __name__ == '__main__':
    main()
