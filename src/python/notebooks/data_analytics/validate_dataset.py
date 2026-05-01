import pandas as pd
import numpy as np
from datetime import datetime
import json

# Load the Excel file
file_path = '/home/ubuntu/Uploads/coffee_shop_sales.xlsx'
print(f"Loading file: {file_path}")
df = pd.read_excel(file_path)

# Basic information
print("\n" + "="*80)
print("DATASET BASIC INFORMATION")
print("="*80)
print(f"Total number of records: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")
print(f"\nColumn names and data types:")
print(df.dtypes)

# Display first few rows
print("\n" + "="*80)
print("FIRST 5 ROWS")
print("="*80)
print(df.head())

# Check for missing values
print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found")

# Date range analysis
print("\n" + "="*80)
print("DATE RANGE ANALYSIS")
print("="*80)
date_col = None
for col in df.columns:
    if 'date' in col.lower():
        date_col = col
        break

if date_col:
    print(f"Date column: {date_col}")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    # Calculate duration
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        duration = (df[date_col].max() - df[date_col].min()).days
        print(f"Duration: {duration} days ({duration/30:.1f} months)")

# Unique values for categorical columns
print("\n" + "="*80)
print("CATEGORICAL VARIABLES - UNIQUE VALUES")
print("="*80)

# Store locations
store_cols = [col for col in df.columns if 'store' in col.lower() or 'location' in col.lower()]
if store_cols:
    for col in store_cols:
        print(f"\n{col}:")
        unique_vals = df[col].unique()
        print(f"  Count: {len(unique_vals)}")
        print(f"  Values: {sorted(unique_vals.tolist())}")

# Product category
product_cat_cols = [col for col in df.columns if 'category' in col.lower()]
if product_cat_cols:
    for col in product_cat_cols:
        print(f"\n{col}:")
        unique_vals = df[col].unique()
        print(f"  Count: {len(unique_vals)}")
        print(f"  Values: {sorted(unique_vals.tolist())}")

# Product type
product_type_cols = [col for col in df.columns if 'type' in col.lower() and 'product' in col.lower()]
if product_type_cols:
    for col in product_type_cols:
        print(f"\n{col}:")
        unique_vals = df[col].unique()
        print(f"  Count: {len(unique_vals)}")
        print(f"  Values: {sorted(unique_vals.tolist())}")

# Product detail
product_detail_cols = [col for col in df.columns if 'detail' in col.lower()]
if product_detail_cols:
    for col in product_detail_cols:
        print(f"\n{col}:")
        unique_vals = df[col].unique()
        print(f"  Count: {len(unique_vals)}")
        print(f"  First 20 values: {sorted(unique_vals.tolist())[:20]}")

# Revenue/sales analysis
print("\n" + "="*80)
print("REVENUE AND SALES ANALYSIS")
print("="*80)

# Look for revenue, price, or sales columns
revenue_cols = [col for col in df.columns if any(term in col.lower() for term in ['revenue', 'price', 'sales', 'total', 'amount'])]
print(f"Revenue-related columns found: {revenue_cols}")

for col in revenue_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        print(f"\n{col}:")
        print(f"  Total: ${df[col].sum():,.2f}")
        print(f"  Mean: ${df[col].mean():.2f}")
        print(f"  Median: ${df[col].median():.2f}")
        print(f"  Min: ${df[col].min():.2f}")
        print(f"  Max: ${df[col].max():.2f}")
        print(f"  Std Dev: ${df[col].std():.2f}")

# Store performance (if revenue data exists and store location exists)
if revenue_cols and store_cols:
    print("\n" + "="*80)
    print("STORE PERFORMANCE COMPARISON")
    print("="*80)
    for store_col in store_cols:
        for revenue_col in revenue_cols:
            if pd.api.types.is_numeric_dtype(df[revenue_col]):
                store_revenue = df.groupby(store_col)[revenue_col].agg(['sum', 'mean', 'count'])
                store_revenue.columns = ['Total Revenue', 'Avg Transaction', 'Transaction Count']
                store_revenue['Total Revenue'] = store_revenue['Total Revenue'].apply(lambda x: f"${x:,.2f}")
                store_revenue['Avg Transaction'] = store_revenue['Avg Transaction'].apply(lambda x: f"${x:.2f}")
                print(f"\n{revenue_col} by {store_col}:")
                print(store_revenue)

# Product category performance
if revenue_cols and product_cat_cols:
    print("\n" + "="*80)
    print("PRODUCT CATEGORY PERFORMANCE")
    print("="*80)
    for cat_col in product_cat_cols:
        for revenue_col in revenue_cols:
            if pd.api.types.is_numeric_dtype(df[revenue_col]):
                cat_revenue = df.groupby(cat_col)[revenue_col].agg(['sum', 'mean', 'count'])
                cat_revenue.columns = ['Total Revenue', 'Avg Transaction', 'Transaction Count']
                cat_revenue = cat_revenue.sort_values('Total Revenue', ascending=False)
                cat_revenue['Total Revenue'] = cat_revenue['Total Revenue'].apply(lambda x: f"${x:,.2f}")
                cat_revenue['Avg Transaction'] = cat_revenue['Avg Transaction'].apply(lambda x: f"${x:.2f}")
                print(f"\n{revenue_col} by {cat_col}:")
                print(cat_revenue)

# Save summary statistics to JSON for report generation
summary_data = {
    "total_records": int(len(df)),
    "columns": df.columns.tolist(),
    "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
    "missing_values": {col: int(count) for col, count in missing.items() if count > 0},
    "date_range": {
        "column": date_col if date_col else None,
        "min": str(df[date_col].min()) if date_col else None,
        "max": str(df[date_col].max()) if date_col else None
    }
}

# Add categorical unique values
if store_cols:
    summary_data["store_locations"] = {col: df[col].unique().tolist() for col in store_cols}
if product_cat_cols:
    summary_data["product_categories"] = {col: df[col].unique().tolist() for col in product_cat_cols}

with open('/home/ubuntu/dataset_summary.json', 'w') as f:
    json.dump(summary_data, f, indent=2, default=str)

print("\n" + "="*80)
print("Summary data saved to: /home/ubuntu/dataset_summary.json")
print("="*80)
