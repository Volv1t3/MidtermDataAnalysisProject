import pandas as pd
import numpy as np

# Load the Excel file
df = pd.read_excel('/home/ubuntu/Uploads/coffee_shop_sales.xlsx')

# Calculate total revenue per transaction
df['total_revenue'] = df['unit_price'] * df['transaction_qty']

print("="*80)
print("CORRECTED REVENUE ANALYSIS (unit_price × transaction_qty)")
print("="*80)
print(f"Total Revenue: ${df['total_revenue'].sum():,.2f}")
print(f"Average Transaction Value: ${df['total_revenue'].mean():.2f}")
print(f"Median Transaction Value: ${df['total_revenue'].median():.2f}")
print(f"Total Units Sold: {df['transaction_qty'].sum():,}")
print(f"Average Units per Transaction: {df['transaction_qty'].mean():.2f}")

print("\n" + "="*80)
print("STORE PERFORMANCE (CORRECTED)")
print("="*80)
store_perf = df.groupby('store_location').agg({
    'total_revenue': 'sum',
    'transaction_id': 'count',
    'transaction_qty': 'sum'
}).round(2)
store_perf.columns = ['Total Revenue', 'Transaction Count', 'Units Sold']
store_perf['Avg Ticket'] = (store_perf['Total Revenue'] / store_perf['Transaction Count']).round(2)
store_perf = store_perf.sort_values('Total Revenue', ascending=False)
print(store_perf)

print("\n" + "="*80)
print("PRODUCT CATEGORY PERFORMANCE (CORRECTED)")
print("="*80)
cat_perf = df.groupby('product_category').agg({
    'total_revenue': 'sum',
    'transaction_id': 'count',
    'transaction_qty': 'sum'
}).round(2)
cat_perf.columns = ['Total Revenue', 'Transaction Count', 'Units Sold']
cat_perf['Avg Transaction Value'] = (cat_perf['Total Revenue'] / cat_perf['Transaction Count']).round(2)
cat_perf['% of Total Revenue'] = (cat_perf['Total Revenue'] / cat_perf['Total Revenue'].sum() * 100).round(1)
cat_perf = cat_perf.sort_values('Total Revenue', ascending=False)
print(cat_perf)

print("\n" + "="*80)
print("TOP 15 PRODUCTS BY REVENUE")
print("="*80)
product_perf = df.groupby('product_detail').agg({
    'total_revenue': 'sum',
    'transaction_id': 'count'
}).round(2)
product_perf.columns = ['Total Revenue', 'Transaction Count']
product_perf = product_perf.sort_values('Total Revenue', ascending=False).head(15)
product_perf['Total Revenue'] = product_perf['Total Revenue'].apply(lambda x: f"${x:,.2f}")
print(product_perf)

print("\n" + "="*80)
print("TIME-BASED ANALYSIS")
print("="*80)
df['month'] = df['transaction_date'].dt.month_name()
df['day_of_week'] = df['transaction_date'].dt.day_name()
df['hour'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S').dt.hour

monthly = df.groupby('month')['total_revenue'].sum().round(2)
print("\nMonthly Revenue:")
month_order = ['January', 'February', 'March', 'April', 'May', 'June']
for month in month_order:
    if month in monthly.index:
        print(f"  {month}: ${monthly[month]:,.2f}")

print("\nDay of Week Performance:")
dow_perf = df.groupby('day_of_week').agg({
    'total_revenue': 'sum',
    'transaction_id': 'count'
}).round(2)
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_perf = dow_perf.reindex(dow_order)
dow_perf.columns = ['Total Revenue', 'Transaction Count']
print(dow_perf)

print("\nPeak Hours (Top 5 by Revenue):")
hourly = df.groupby('hour')['total_revenue'].sum().sort_values(ascending=False).head(5)
for hour, revenue in hourly.items():
    print(f"  {hour:02d}:00 - {hour+1:02d}:00: ${revenue:,.2f}")

print("\n" + "="*80)
print("QUANTITY ANALYSIS")
print("="*80)
print(f"Transactions with 1 item: {len(df[df['transaction_qty'] == 1]):,} ({len(df[df['transaction_qty'] == 1])/len(df)*100:.1f}%)")
print(f"Transactions with 2+ items: {len(df[df['transaction_qty'] > 1]):,} ({len(df[df['transaction_qty'] > 1])/len(df)*100:.1f}%)")
print(f"Max quantity in single transaction: {df['transaction_qty'].max()}")

print("\n" + "="*80)
print("COMPLETE UNIQUE VALUES LIST")
print("="*80)
print(f"\nAll Product Details ({len(df['product_detail'].unique())} unique):")
for i, product in enumerate(sorted(df['product_detail'].unique()), 1):
    print(f"  {i}. {product}")

