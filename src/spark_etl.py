from pyspark.sql import SparkSession
from pyspark.sql.functions import col, input_file_name, regexp_extract, lag, when, lit, to_date
from pyspark.sql.window import Window

def main():
    # Initialize Spark - robust config for Windows/Java 22+
    spark = SparkSession.builder \
        .appName("EquityPro_ETL") \
        .master("local[1]") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print("Loading Technical Data via Spark...")
    
    # 1. Ingest
    # Recursive load might handle it, or just wildcard. 
    # Validated path: dataset/technicals_csv/*.csv
    raw_df = spark.read.option("header", "true") \
                 .option("inferSchema", "true") \
                 .csv("dataset/technicals_csv/*.csv") \
                 .withColumn("filename", input_file_name())

    # 2. Cleanup & Ticker Extraction
    # Handle misnamed Date column (Price -> Date)
    if "Price" in raw_df.columns:
        raw_df = raw_df.withColumnRenamed("Price", "Date")
    
    # Extract Ticker from filename: .../ABB_technicals.csv -> ABB
    # Adjust regex for Windows paths if needed, or generic slash
    df = raw_df.withColumn("Ticker", regexp_extract("filename", r"([^\/\\]+)_technicals\.csv$", 1))

    # 3. Feature Engineering
    # Cast Columns
    numeric_cols = ["Close", "Open", "High", "Low", "50_SMA", "200_SMA"]
    for c in numeric_cols:
        if c in df.columns:
            df = df.withColumn(c, col(c).cast("float"))
        else:
            df = df.withColumn(c, lit(0.0))

    # Parse Date (assuming format is suitable or inferred)
    # If inferred as string, we might need to_date(col("Date"))
    # The pandas check showed "2016-03-31", should be auto-inferred or standard.
    
    # Window for Lag calculations
    w = Window.partitionBy("Ticker").orderBy("Date")

    # Features:
    # Target: Next Close > Close
    # Bullish_MA: 50 > 200
    # Risk_Score: (High - Low) / Open * 100
    df = df.withColumn("NextClose", lag("Close", -1).over(w)) \
           .withColumn("Target", (col("NextClose") > col("Close")).cast("int")) \
           .withColumn("Bullish_MA", (col("50_SMA") > col("200_SMA")).cast("int")) \
           .withColumn("Volatility", (col("High") - col("Low")) / col("Open")) \
           .withColumn("Risk_Score", col("Volatility") * 100)

    # 4. Save
    # Drop rows with null targets (last row per group) or vital features
    final_df = df.dropna(subset=["Target", "Risk_Score", "Close"]) \
                 .select("Ticker", "Date", "Close", "Open", "High", "Low", "Volume", 
                         "Risk_Score", "Bullish_MA", "Target", "Volatility")
    
    output_path = "dataset/spark_processed/processed_data.parquet"
    print(f"Saving processed data to {output_path}...")
    
    # Coalesce to 1 to produce fewer files (cleaner for local usage)
    final_df.coalesce(1).write.mode("overwrite").parquet("dataset/spark_processed")
    
    print("Done.")
    spark.stop()

if __name__ == "__main__":
    main()
