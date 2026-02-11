"""Kafka to S3 Streaming Job

This Spark Streaming job reads events from Kafka, parses them using
the Spark parser, and writes them to S3 in Parquet format partitioned
by date and event_type.
"""

import os
import logging
from pyspark.sql import SparkSession
from parsers.spark_parser import parse_kafka_stream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_spark_session() -> SparkSession:
    """Create and configure Spark session for AWS S3 access"""
    # AWS S3 configuration
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    builder = SparkSession.builder \
        .appName("ShrekDataLake-KafkaToS3") \
        .config("spark.sql.streaming.schemaInference", "false") \
        .config("spark.streaming.stopGracefullyOnShutdown", "true")

    builder = builder \
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key) \
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{aws_region}.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")

    return builder.getOrCreate()


def read_kafka_stream(spark: SparkSession) -> "DataFrame":
    """Read streaming data from Kafka"""
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_topic = os.getenv("KAFKA_TOPIC", "movielog11")
    starting_offsets = os.getenv("SPARK_STARTING_OFFSETS", "latest")

    logger.info(f"Connecting to Kafka: {kafka_bootstrap}, topic: {kafka_topic}")

    return spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap) \
        .option("subscribe", kafka_topic) \
        .option("startingOffsets", starting_offsets) \
        .option("failOnDataLoss", "false") \
        .load()


def write_to_s3(df: "DataFrame") -> "StreamingQuery":
    """Write parsed events to S3 in Parquet format"""
    s3_bucket = os.getenv("S3_BUCKET", "shrek-events")
    s3_path = f"s3a://{s3_bucket}/events/"
    checkpoint_location = os.getenv("CHECKPOINT_LOCATION", "/tmp/spark-checkpoint")
    trigger_interval = os.getenv("TRIGGER_INTERVAL", "30 seconds")

    logger.info(f"Writing to S3: {s3_path}")
    logger.info(f"Checkpoint location: {checkpoint_location}")
    logger.info(f"Trigger interval: {trigger_interval}")

    return df.writeStream \
        .format("parquet") \
        .option("path", s3_path) \
        .option("checkpointLocation", checkpoint_location) \
        .partitionBy("date", "event_type") \
        .outputMode("append") \
        .trigger(processingTime=trigger_interval) \
        .start()


def main():
    """Main entry point for the streaming job"""
    logger.info("Starting Kafka to S3 streaming job...")

    # Create Spark session
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("INFO")

    logger.info(f"Spark version: {spark.version}")

    try:
        # Read from Kafka
        kafka_stream = read_kafka_stream(spark)

        # Parse events
        logger.info("Parsing events...")
        parsed_stream = parse_kafka_stream(kafka_stream)

        # Show schema for debugging
        logger.info("Parsed stream schema:")
        parsed_stream.printSchema()

        # Write to S3
        query = write_to_s3(parsed_stream)

        logger.info("Streaming query started. Waiting for data...")
        logger.info(f"Query ID: {query.id}")
        logger.info(f"Query name: {query.name}")

        # Wait for termination
        query.awaitTermination()

    except KeyboardInterrupt:
        logger.info("Received shutdown signal, stopping gracefully...")
        spark.streams.active[0].stop()
    except Exception as e:
        logger.error(f"Error in streaming job: {e}", exc_info=True)
        raise
    finally:
        logger.info("Stopping Spark session...")
        spark.stop()


if __name__ == "__main__":
    main()
