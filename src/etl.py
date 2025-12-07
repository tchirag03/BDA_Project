from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lit, avg, when
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
import os
import glob

def create_spark_session():
    return SparkSession.builder \
        .appName("EquityAnalysisETL") \
        .config("spark.sql.warehouse.dir", "database") \
        .getOrCreate()

def process_fundamentals(spark, input_dir, output_path):
    print("Processing Fundamentals...")
    # Define schema based on inspection
    # Expected columns in cleaned CSV: Company, Period, ... (rest from original)
    # We need to map them to: Ticker, Date, NetProfit, Sales, OperatingProfit, EquityShareCapital, Reserves, Borrowings, Debt, NetWorth, EBIT, ROE, ROCE, EPS, BVPS
    
    # Since the column names in CSV might vary or be "Net profit", "Sales", etc., we will read with header=True and infer schema first, then select/rename.
    
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print("No fundamental files found.")
        return

    # We'll read all CSVs. Since they have the same structure (cleaned), we can read them together or iterate.
    # Iterating is safer to handle filename -> Ticker mapping if needed, but 'Company' column is already there.
    
    # Let's read one to check schema if needed, but for now we trust the cleaning.
    # The cleaned files have 'Company' and 'Period'. We need to map 'Period' to a Date or keep it as is?
    # The PRD requires 'Date'. The fundamental data didn't have explicit dates, just 'Period'.
    # We will assume 'Period' 1 is the latest year, say 2024, and decrement. 
    # Or better, just store 'Period' as 'Date' string for now if actual date is missing.
    # Wait, the user wants "Date". I'll synthesize a date based on Period (e.g., Period 1 = 2024-03-31).
    
    # Schema for reading (inferring is easier for now as columns are many)
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_dir)
    
    # Rename columns to match PRD Schema
    # Mapping: 
    # Company -> Ticker
    # Net profit -> NetProfit
    # Sales -> Sales
    # ... need to check exact CSV headers. 
    # Based on previous `view_file` of `dataset/fundamentals_cleaned/Infosys.csv`:
    # Headers: Company, Period, No., Name, Sales, ...
    
    # Let's normalize column names
    # We need to select specific columns.
    
    # PRD Schema: Ticker, Date, NetProfit, Sales, OperatingProfit, EquityShareCapital, Reserves, Borrowings, Debt, NetWorth, EBIT, ROE, ROCE, EPS, BVPS
    
    # We need to do some column mapping. 
    # Let's try to select and rename.
    
    # Note: Spark is case insensitive by default for column selection usually.
    
    df_transformed = df.withColumnRenamed("Company", "Ticker") \
                       .withColumnRenamed("Net profit", "NetProfit") \
                       .withColumnRenamed("Operating Profit", "OperatingProfit") \
                       .withColumnRenamed("Equity Capital", "EquityShareCapital") \
                       .withColumnRenamed("Total Liabilities", "Debt") \
                       .withColumnRenamed("Total Assets", "NetWorth") # Approximation if NetWorth not explicit
                       # Add other mappings as found in the CSV
    
    # For Date: Create a dummy date based on Period. 
    # Assuming Period 1 is 2024, 2 is 2023...
    # Period is likely a string or int.
    df_transformed = df_transformed.withColumn("Year", 2025 - col("Period")) \
                                   .withColumn("Date", to_date(col("Year").cast("string"), "yyyy"))
    
    # Select only required columns (if they exist)
    # We will select available columns and fill missing with null or 0
    required_cols = ["Ticker", "Date", "NetProfit", "Sales", "OperatingProfit", "EquityShareCapital", "Reserves", "Borrowings", "Debt", "NetWorth", "EBIT", "ROE", "ROCE", "EPS", "BVPS"]
    
    selected_cols = []
    for c in required_cols:
        if c in df_transformed.columns:
            selected_cols.append(col(c))
        else:
            selected_cols.append(lit(None).alias(c))
            
    final_df = df_transformed.select(*selected_cols)
    
    # Write to Parquet
    final_df.write.mode("overwrite").parquet(output_path)
    print(f"Fundamentals saved to {output_path}")
    
    # Register Table
    final_df.createOrReplaceTempView("fundamentals")
    spark.sql(f"CREATE TABLE IF NOT EXISTS fundamentals USING PARQUET LOCATION '{output_path}'")

def process_technicals(spark, input_dir, output_path):
    print("Processing Technicals...")
    
    # Schema: Ticker, Date, Close, High, Low, Open, Volume, 20_SMA, 50_SMA
    # We need to calculate SMA200 and DailyReturn
    
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_dir)
    
    # Rename columns to match PRD
    # Company -> Ticker
    df = df.withColumnRenamed("Company", "Ticker")
    
    # Ensure Date is DateType
    df = df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd"))
    
    # Calculate Daily Return: (Close - PrevClose) / PrevClose
    # Using Window function
    from pyspark.sql.window import Window
    from pyspark.sql.functions import lag
    
    windowSpec = Window.partitionBy("Ticker").orderBy("Date")
    df = df.withColumn("PrevClose", lag("Close").over(windowSpec))
    df = df.withColumn("DailyReturn", (col("Close") - col("PrevClose")) / col("PrevClose"))
    
    # Calculate SMA200 (if not present, or re-calculate to be sure)
    # The CSV has 20_SMA and 50_SMA.
    df = df.withColumnRenamed("20_SMA", "SMA20").withColumnRenamed("50_SMA", "SMA50")
    
    # Calculate SMA200 using window
    windowSMA200 = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-199, 0)
    df = df.withColumn("SMA200", avg("Close").over(windowSMA200))
    
    # Select required columns
    # PRD: Ticker, Date, Open, High, Low, Close, Volume, DailyReturn, SMA20, SMA50, SMA200, Label, Technical_ML_Score, TrendScore, TechnicalScore
    # Label and Scores will be added by ML/Scoring later. We initialize them as null.
    
    df_final = df.select(
        col("Ticker"), col("Date"), col("Open"), col("High"), col("Low"), col("Close"), col("Volume"),
        col("DailyReturn"), col("SMA20"), col("SMA50"), col("SMA200"),
        lit(None).cast(StringType()).alias("Label"),
        lit(None).cast(DoubleType()).alias("Technical_ML_Score"),
        lit(None).cast(DoubleType()).alias("TrendScore"),
        lit(None).cast(DoubleType()).alias("TechnicalScore")
    )
    
    # Write to Parquet
    df_final.write.mode("overwrite").parquet(output_path)
    print(f"Technicals saved to {output_path}")
    
    # Register Table
    df_final.createOrReplaceTempView("technicals")
    spark.sql(f"CREATE TABLE IF NOT EXISTS technicals USING PARQUET LOCATION '{output_path}'")

def main():
    spark = create_spark_session()
    
    # Paths
    fund_input = "dataset/fundamentals_cleaned"
    tech_input = "dataset/technicals_cleaned"
    
    fund_output = "database/fundamental_db"
    tech_output = "database/technical_db"
    
    # Process
    process_fundamentals(spark, fund_input, fund_output)
    process_technicals(spark, tech_input, tech_output)
    
    spark.stop()

if __name__ == "__main__":
    main()
