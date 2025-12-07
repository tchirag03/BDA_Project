import os
import pandas as pd
import glob

def clean_technicals():
    # Define paths
    input_dir = r"dataset/technicals_csv"
    output_dir = r"dataset/technicals_cleaned"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"Found {len(csv_files)} files to process.")
    
    # Standard headers
    headers = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', '20_SMA', '50_SMA']
    
    for file_path in csv_files:
        try:
            # Get filename for Company column
            filename = os.path.basename(file_path)
            # Extract company name (e.g., ADANIENT_technicals.csv -> ADANIENT)
            company_name = filename.replace("_technicals.csv", "")
            
            # Read CSV, skipping the first 3 rows (garbage headers)
            # The data starts from row 4 (index 3)
            df = pd.read_csv(file_path, skiprows=3, header=None)
            
            # Assign standard headers
            # Note: The file might have extra columns or fewer columns, we need to be careful
            # Based on inspection, there are 8 columns corresponding to the headers
            if len(df.columns) == 8:
                df.columns = headers
            else:
                # If column count doesn't match, we might need to adjust or log a warning
                # For now, let's try to assign first 8 and drop rest if any, or pad
                print(f"Warning: {filename} has {len(df.columns)} columns. Expected 8.")
                if len(df.columns) > 8:
                    df = df.iloc[:, :8]
                    df.columns = headers
                else:
                    # If fewer, we can't easily map without knowing which are missing
                    print(f"Skipping {filename} due to column mismatch.")
                    continue

            # Add metadata columns
            df.insert(0, "Company", company_name)
            
            # Save cleaned file
            output_path = os.path.join(output_dir, filename)
            df.to_csv(output_path, index=False)
            print(f"Processed: {filename}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    clean_technicals()
