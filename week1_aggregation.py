import pandas as pd
import glob
from pathlib import Path

# Week 1: Monthly Dataset Aggregation
# Combine available monthly MLS datasets starting January 2024.

data_folder = Path("csv")

sold_files = sorted(glob.glob(str(data_folder / "CRMLSSold*.csv")))
listing_files = sorted(glob.glob(str(data_folder / "CRMLSListing*.csv")))

print("Sold files found:", len(sold_files))
print("Listing files found:", len(listing_files))

# Load monthly sold datasets
sold_dfs = []

for file in sold_files:
    df = pd.read_csv(file, low_memory=False)
    print(f"{Path(file).name}: {len(df)} rows")
    sold_dfs.append(df)

if not sold_dfs:
    raise ValueError("No sold CSV files found.")

# Count rows before concatenation
sold_rows_before = sum(len(df) for df in sold_dfs)
print("\nSold rows before concatenation:", sold_rows_before)

# Combine monthly datasets
sold = pd.concat(sold_dfs, ignore_index=True)
print("Sold rows after concatenation:", len(sold))

# Filter to Residential properties
print("Sold rows before Residential filter:", len(sold))

sold = sold[sold["PropertyType"] == "Residential"].copy()

print("Sold rows after Residential filter:", len(sold))

# Save combined dataset
sold.to_csv("combined_sold_residential.csv", index=False)

print("Sold dataset saved successfully!")


# Load monthly listing datasets
listing_dfs = []

for file in listing_files:
    df = pd.read_csv(file, low_memory=False)
    print(f"{Path(file).name}: {len(df)} rows")
    listing_dfs.append(df)

if not listing_dfs:
    raise ValueError("No listing CSV files found.")

# Count rows before concatenation
listing_rows_before = sum(len(df) for df in listing_dfs)
print("\nListing rows before concatenation:", listing_rows_before)

# Combine monthly datasets
listings = pd.concat(listing_dfs, ignore_index=True)
print("Listing rows after concatenation:", len(listings))

# Filter to Residential properties
print("Listing rows before Residential filter:", len(listings))

listings = listings[
    listings["PropertyType"] == "Residential"
].copy()

print("Listing rows after Residential filter:", len(listings))

# Save combined dataset
listings.to_csv("combined_listings_residential.csv", index=False)

print("Listing dataset saved successfully!")

