from pathlib import Path
import pandas as pd
import os

# Check files
data_dir = Path("../data")
print("Files in data directory:")
for file in data_dir.iterdir():
	print(f"	{file.name}")

# Load main dataset, attempt to locate CSV files
csv_files = list(data_dir.glob("*.csv"))
if csv_files:
	df = pd.read_csv(csv_files[0])
	print(f"\nLoaded: {csv_files[0].name}")
	print(f"Shape: {df.shape}")
	print(f"\nColumns: {df.columns.tolist()}")
	print(f"\nFirst few rows:")
	print(df.head())
	print(f"\nData Types:")
	print(df.dtypes)
	print(f"\nMissing values:")
	print(df.isnull().sum())
