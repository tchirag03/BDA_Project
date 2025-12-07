import os
import pandas as pd
import glob
import numpy as np

def clean_fundamentals():
    # Define paths
    input_dir = r"dataset/fundamental2"
    output_dir = r"dataset/fundamentals_cleaned"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"Found {len(csv_files)} files to process in {input_dir}.")
    
    for file_path in csv_files:
        try:
            # Get filename for Company column
            filename = os.path.basename(file_path)
            company_name = os.path.splitext(filename)[0]
            
            # Read CSV without header to handle transposition manually
            df = pd.read_csv(file_path, header=None)
            
            # Transpose the dataframe
            df_transposed = df.T
            
            # Use the first row as header
            new_header = df_transposed.iloc[0]
            df_cleaned = df_transposed[1:]
            df_cleaned.columns = new_header
            
            # Rename 'Report Date' to 'Date' if present, or ensure a Date column exists
            # Common patterns for the date row: 'Report Date', 'Unnamed: 0' (if empty in original), etc.
            # In the inspected file, it was 'Report Date'.
            
            # Strip whitespace from column names
            df_cleaned.columns = df_cleaned.columns.str.strip()
            
            if 'Report Date' in df_cleaned.columns:
                df_cleaned = df_cleaned.rename(columns={'Report Date': 'Date'})
            
            # Normalize column names (optional, but helpful for ETL)
            # Remove special chars, extra spaces from column names
            # But let's verify what the ETL expects. ETL expects 'Net profit', 'Sales' etc.
            # The new file has 'Net profit', 'Sales', etc. so we should be good.
            
            # Insert Company name
            df_cleaned.insert(0, "Company", company_name)
            
            # Remove rows where Date might be null (if any artifacts from transposition)
            if 'Date' in df_cleaned.columns:
                df_cleaned = df_cleaned.dropna(subset=['Date'])
            
            # Save cleaned file
            output_path = os.path.join(output_dir, filename) # Keep same filename
            df_cleaned.to_csv(output_path, index=False)
            print(f"Processed: {filename}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    clean_fundamentals()
