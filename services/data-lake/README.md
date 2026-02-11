# Data Lake Service

Kafka to S3 streaming ingestion service using Spark Structured Streaming.

## Architecture

This service reads events from Kafka, parses them using the shared schema definitions, and writes them to AWS S3 in Parquet format.

```
Kafka (External) → Spark Streaming → AWS S3 (External)
                         ↓
              Parsing Rules (Shared)
              UnifiedEventRecord Schema
```

## Features

- **Single Source of Truth**: Event schemas and parsing rules defined once in `shared/shrekommender_common`
- **Spark SQL Native Parsing**: High-performance parsing using Spark SQL functions (no Python UDF overhead)
- **Partitioned Storage**: Data partitioned by `date` and `event_type` for efficient querying
- **Automatic Schema**: Spark schema auto-generated from Pydantic models

## Setup

### 1. Install Dependencies

```bash
cd services/data-lake
uv sync
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required variables:
- `KAFKA_BOOTSTRAP_SERVERS`: Your Kafka cluster endpoint
- `KAFKA_TOPIC`: Kafka topic to consume from
- `S3_BUCKET`: AWS S3 bucket name
- `AWS_ACCESS_KEY_ID`: AWS credentials
- `AWS_SECRET_ACCESS_KEY`: AWS credentials

### 3. Run Locally

```bash
python spark_jobs/kafka_to_s3.py
```

### 4. Run with Docker Compose

From project root:

```bash
cd infrastructure
cp .env.example .env  # Configure your external dependencies
docker-compose up data-lake
```

## Output Schema

Data is written to S3 with the following structure:

```
s3://your-bucket/events/
  date=2025-10-25/
    event_type=watch/
      part-00000.parquet
    event_type=rate/
      part-00000.parquet
    event_type=recommendation/
      part-00000.parquet
```

### Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| event_id | string | Unique event ID (UUID) |
| timestamp | timestamp | Event timestamp |
| date | string | Event date (partition key) |
| user_id | string | User ID |
| event_type | string | watch / rate / recommendation |
| movie_id | string | Movie ID (watch & rate events) |
| minute | int | Minute watched (watch events) |
| rating | int | Rating 1-5 (rate events) |
| server | string | Server name (recommendation events) |
| status_code | int | HTTP status (recommendation events) |
| recommendations | string | Recommended IDs (recommendation events) |
| response_time | string | Response time (recommendation events) |
| ingestion_time | timestamp | Data ingestion timestamp |
| source_topic | string | Source Kafka topic |

## Development

### Modify Parsing Logic

Parsing rules are defined in `shared/shrekommender_common/schemas/parsing_rules.py`:

```python
PARSING_RULES = [
    ParsingRule(
        event_type="watch",
        pattern=r'GET /data/m/([^/]+)/(\d+)\.mpg',
        field_mappings={"movie_id": 1, "minute": 2}
    ),
    # ...
]
```

After modifying, the Spark parser automatically uses the new rules.

### Add New Event Type

1. Add Pydantic model in `shared/shrekommender_common/schemas/events.py`
2. Add parsing rule in `shared/shrekommender_common/schemas/parsing_rules.py`
3. Update `UnifiedEventRecord` schema with new fields
4. Spark schema is auto-generated, no manual updates needed!

## Monitoring

Check Spark UI at `http://localhost:4040` when running locally.

View streaming query progress in logs:
```bash
docker-compose logs -f data-lake
```
