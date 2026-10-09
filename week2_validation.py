
import pandas as pd

# Week 2: Dataset Structuring and Validation
# Analyze the Residential MLS datasets created in Week 1.

# Load the combined datasets
sold = pd.read_csv("combined_sold_residential.csv", low_memory=False)
listings = pd.read_csv("combined_listings_residential.csv", low_memory=False)

print("Sold dataset shape:", sold.shape)
print("Listings dataset shape:", listings.shape)

# Confirm Residential filtering
print("\nSold property types:")
print(sold["PropertyType"].value_counts(dropna=False))

print("\nListing property types:")
print(listings["PropertyType"].value_counts(dropna=False))

# Calculate missing-value counts and percentages
def missing_value_summary(df):
    summary = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing Percentage": df.isnull().mean() * 100
    })
    return summary.sort_values("Missing Percentage", ascending=False)

sold_missing = missing_value_summary(sold)
listings_missing = missing_value_summary(listings)

print("\nSold missing-value summary:")
print(sold_missing)

print("\nListings missing-value summary:")
print(listings_missing)

# Flag columns with more than 90% missing values
print("\nSold columns with over 90% missing values:")
print(sold_missing[sold_missing["Missing Percentage"] > 90])

print("\nListing columns with over 90% missing values:")
print(listings_missing[listings_missing["Missing Percentage"] > 90])

# Summarize numeric distributions
numeric_columns = ["ClosePrice", "LivingArea", "DaysOnMarket"]

print("\nSold numeric summary:")
print(sold[numeric_columns].describe())

print("\nListings numeric summary:")
print(listings[numeric_columns].describe())

# Save analysis summaries locally
sold_missing.to_csv("sold_missing_summary.csv")
listings_missing.to_csv("listings_missing_summary.csv")

sold[numeric_columns].describe().to_csv("sold_numeric_summary.csv")
listings[numeric_columns].describe().to_csv("listings_numeric_summary.csv")

print("\nWeek 2 validation completed!")
