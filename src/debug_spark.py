from pyspark.sql import SparkSession

def test_spark():
    try:
        print("Initializing SparkSession...")
        spark = SparkSession.builder \
            .appName("TestSpark") \
            .master("local[*]") \
            .getOrCreate()
        print("SparkSession created successfully.")
        
        data = [("Alice", 1), ("Bob", 2)]
        df = spark.createDataFrame(data, ["Name", "Value"])
        df.show()
        print("DataFrame operation successful.")
        
        spark.stop()
    except Exception as e:
        print(f"Spark Error: {e}")

if __name__ == "__main__":
    test_spark()
