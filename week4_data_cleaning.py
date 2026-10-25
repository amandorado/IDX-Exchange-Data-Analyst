import pandas as pd
import numpy as np

# Load datasets from Week 3
sold = pd.read_csv(
    '/Users/amandorado/Desktop/IDX-Exchange-Data-Analyst/combined_sold_with_rates.csv',
    low_memory=False
)

listings = pd.read_csv(
    '/Users/amandorado/Desktop/IDX-Exchange-Data-Analyst/combined_listings_with_rates.csv',
    low_memory=False
)
# Check original dataset sizes
print("Sold dataset:", sold.shape)
print("Listings dataset:", listings.shape)

# Review column names and data types
print("\nSold column types:")
print(sold.dtypes)

print("\nListings column types:")
print(listings.dtypes)

# Convert date columns to datetime format
date_columns = [
    'CloseDate',
    'PurchaseContractDate',
    'ListingContractDate',
    'ContractStatusChangeDate'
]

for df in [sold, listings]:
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

# Confirm date conversions
print("\nSold date column types:")
print(sold[[col for col in date_columns if col in sold.columns]].dtypes)

print("\nListings date column types:")
print(listings[[col for col in date_columns if col in listings.columns]].dtypes)

# Step 3: Convert numeric columns and flag invalid values

numeric_columns = [
    'ClosePrice',
    'ListPrice',
    'OriginalListPrice',
    'LivingArea',
    'DaysOnMarket',
    'BedroomsTotal',
    'BathroomsTotalInteger',
    'Latitude',
    'Longitude'
]

for name, df in [('Sold', sold), ('Listings', listings)]:

    # Convert numeric columns
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Identify invalid values
    invalid_rules = {
        'ClosePrice': lambda x: x <= 0,
        'LivingArea': lambda x: x <= 0,
        'DaysOnMarket': lambda x: x < 0,
        'BedroomsTotal': lambda x: x < 0,
        'BathroomsTotalInteger': lambda x: x < 0
    }

    print(f"\n{name} invalid numeric values:")

    for col, rule in invalid_rules.items():
        if col in df.columns:
            count = rule(df[col]).sum()
            print(f"{col}: {count:,} invalid values")

# Step 4: Flag invalid numeric values

for df in [sold, listings]:
    df['invalid_close_price_flag'] = df['ClosePrice'] <= 0
    df['invalid_living_area_flag'] = df['LivingArea'] <= 0
    df['invalid_days_on_market_flag'] = df['DaysOnMarket'] < 0
    df['invalid_bedrooms_flag'] = df['BedroomsTotal'] < 0
    df['invalid_bathrooms_flag'] = df['BathroomsTotalInteger'] < 0

# Count records with at least one invalid numeric value
flag_columns = [
    'invalid_close_price_flag',
    'invalid_living_area_flag',
    'invalid_days_on_market_flag',
    'invalid_bedrooms_flag',
    'invalid_bathrooms_flag'
]

for name, df in [('Sold', sold), ('Listings', listings)]:
    invalid_rows = df[flag_columns].any(axis=1).sum()

    print(f"\n{name} records with invalid numeric values: {invalid_rows:,}")

# Step 5: Analyze missing values

for name, df in [('Sold', sold), ('Listings', listings)]:

    # Calculate missing counts and percentages
    missing_report = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing Percentage': df.isnull().mean() * 100
    })

    # Sort columns by highest missing percentage
    missing_report = missing_report.sort_values(
        by='Missing Percentage',
        ascending=False
    )

    print(f"\n{name} columns with more than 90% missing values:")

    high_missing = missing_report[
        missing_report['Missing Percentage'] > 90
    ]

    print(high_missing)

    # Save the complete missing value report
    missing_report.to_csv(
        f'{name.lower()}_missing_report.csv'
    )

# Save Week 4 datasets with validation flags
sold.to_csv('week4_sold_flagged.csv', index=False)
listings.to_csv('week4_listings_flagged.csv', index=False)

print("\nWeek 4 datasets saved successfully!")