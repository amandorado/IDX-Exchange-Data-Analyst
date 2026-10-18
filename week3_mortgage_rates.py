"""Week 3: Add monthly FRED 30-year fixed mortgage rates to residential MLS data."""
import pandas as pd

# Load the Residential-only datasets produced in Week 1.
sold = pd.read_csv('combined_sold_residential.csv', low_memory=False)
listings = pd.read_csv('combined_listings_residential.csv', low_memory=False)

# Fetch weekly Freddie Mac mortgage rates from FRED (no API key required).
url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US'
mortgage = pd.read_csv(url)
mortgage.columns = ['date', 'rate_30yr_fixed']
mortgage['date'] = pd.to_datetime(mortgage['date'], errors='coerce')
mortgage['rate_30yr_fixed'] = pd.to_numeric(mortgage['rate_30yr_fixed'], errors='coerce')

# Average the weekly observations within each calendar month.
mortgage['year_month'] = mortgage['date'].dt.to_period('M')
mortgage_monthly = (mortgage.groupby('year_month', as_index=False)['rate_30yr_fixed'].mean())

# Align each MLS record to the month of its relevant date.
sold['year_month'] = pd.to_datetime(sold['CloseDate'], errors='coerce').dt.to_period('M')
listings['year_month'] = pd.to_datetime(listings['ListingContractDate'], errors='coerce').dt.to_period('M')

# Left joins preserve every original property record.
sold_with_rates = sold.merge(mortgage_monthly, on='year_month', how='left', validate='many_to_one')
listings_with_rates = listings.merge(mortgage_monthly, on='year_month', how='left', validate='many_to_one')

# Validate both row counts and mortgage-rate coverage.
for name, original, enriched in [
    ('Sold', sold, sold_with_rates),
    ('Listings', listings, listings_with_rates),
]:
    missing = enriched['rate_30yr_fixed'].isna().sum()
    print(f'{name}: {len(original):,} rows before, {len(enriched):,} after merge')
    print(f'{name}: {missing:,} rows without mortgage rates')
    if len(original) != len(enriched):
        raise ValueError(f'{name}: merge changed the number of rows')
    if missing:
        print(f'{name}: unmatched months / missing dates:')
        print(enriched.loc[enriched['rate_30yr_fixed'].isna(), 'year_month'].value_counts(dropna=False).head(15))

# Save locally; MLS datasets are confidential and should not be pushed to a public repository.
sold_with_rates.to_csv('combined_sold_with_rates.csv', index=False)
listings_with_rates.to_csv('combined_listings_with_rates.csv', index=False)
print('Week 3 enriched datasets saved.')
