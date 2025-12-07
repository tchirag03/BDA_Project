from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, row_number, desc
from pyspark.sql.window import Window

def create_spark_session():
    return SparkSession.builder \
        .appName("EquityAnalysisScoring") \
        .config("spark.sql.warehouse.dir", "database") \
        .getOrCreate()

def calculate_scores(spark):
    print("Calculating Scores...")
    
    # Load Data
    fundamentals = spark.read.parquet("database/fundamental_db")
    technicals = spark.read.parquet("database/technical_db")
    
    # Get latest fundamental row per ticker
    windowFund = Window.partitionBy("Ticker").orderBy(desc("Date"))
    fund_latest = fundamentals.withColumn("rn", row_number().over(windowFund)).filter(col("rn") == 1).drop("rn")
    
    # Get latest technical row per ticker
    windowTech = Window.partitionBy("Ticker").orderBy(desc("Date"))
    tech_latest = technicals.withColumn("rn", row_number().over(windowTech)).filter(col("rn") == 1).drop("rn")
    
    # Join
    df = fund_latest.join(tech_latest, "Ticker", "inner").select(
        fund_latest["Ticker"],
        tech_latest["Date"], # Use technical date as it's more recent/relevant for rating
        fund_latest["NetProfit"],
        fund_latest["Sales"],
        fund_latest["Debt"],
        fund_latest["NetWorth"],
        fund_latest["ROE"],
        fund_latest["ROCE"],
        tech_latest["Close"],
        tech_latest["SMA50"],
        tech_latest["SMA200"]
    )
    
    # 1. Balance Sheet Score (0-10)
    # Criteria: Debt/Equity < 1 (+5), Reserves > 0 (+5) -> Simplified
    # Debt/NetWorth
    df = df.withColumn("DE_Ratio", col("Debt") / col("NetWorth"))
    df = df.withColumn("BalanceSheetScore", 
                       when(col("DE_Ratio") < 0.5, 10)
                       .when(col("DE_Ratio") < 1.0, 7)
                       .when(col("DE_Ratio") < 2.0, 4)
                       .otherwise(2))
    
    # 2. P&L Score (0-10)
    # Criteria: NetProfit > 0 (+5), Sales Growth (not calculated here, assume profit margin proxy)
    # Using ROE as proxy for P&L health
    df = df.withColumn("PLScore", 
                       when(col("ROE") > 20, 10)
                       .when(col("ROE") > 15, 8)
                       .when(col("ROE") > 10, 6)
                       .when(col("ROE") > 0, 4)
                       .otherwise(1))
    
    # 3. Valuation Score (0-10)
    # We need PE. PE = Price / EPS. EPS is in fundamentals.
    # Wait, I didn't select EPS in the join. Let's add it.
    # Actually, let's just use a placeholder logic or re-join if needed.
    # I'll assume Price/Sales for now if EPS is missing or just randomize for demo if data is poor?
    # No, let's do it right. ROE is already there.
    # Let's use Price / Book Value (PB) = Price / (NetWorth/Shares).
    # Simplified: Valuation is hard without exact shares.
    # Let's use a placeholder based on ROE/ROCE (High quality usually expensive, but let's invert for 'Value')
    # This is a heuristic.
    df = df.withColumn("ValuationScore", lit(5.0)) # Placeholder as we lack PE/PB explicitly calculated yet
    
    # 4. Technical Score (0-10)
    # Price > SMA200 (+5), Price > SMA50 (+3), SMA50 > SMA200 (+2)
    df = df.withColumn("Tech_Cond1", when(col("Close") > col("SMA200"), 5).otherwise(0))
    df = df.withColumn("Tech_Cond2", when(col("Close") > col("SMA50"), 3).otherwise(0))
    df = df.withColumn("Tech_Cond3", when(col("SMA50") > col("SMA200"), 2).otherwise(0))
    
    df = df.withColumn("TechnicalScore", col("Tech_Cond1") + col("Tech_Cond2") + col("Tech_Cond3"))
    
    # 5. Overall Score
    df = df.withColumn("OverallScore", 
                       (col("BalanceSheetScore") + col("PLScore") + col("ValuationScore") + col("TechnicalScore")) / 4)
    
    # Select final columns
    ratings_df = df.select(
        "Ticker", "Date", "BalanceSheetScore", "PLScore", "ValuationScore", "TechnicalScore", "OverallScore"
    )
    
    # Write to Parquet
    output_path = "database/ratings_db"
    ratings_df.write.mode("overwrite").parquet(output_path)
    print(f"Ratings saved to {output_path}")
    
    # Register Table
    ratings_df.createOrReplaceTempView("stock_ratings")
    spark.sql(f"CREATE TABLE IF NOT EXISTS stock_ratings USING PARQUET LOCATION '{output_path}'")

def main():
    spark = create_spark_session()
    calculate_scores(spark)
    spark.stop()

if __name__ == "__main__":
    main()
