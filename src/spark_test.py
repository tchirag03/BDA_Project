from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Test").master("local[*]").config("spark.driver.bindAddress", "127.0.0.1").getOrCreate()
print("Spark Alive")
spark.sql("SELECT 1").show()
spark.stop()
