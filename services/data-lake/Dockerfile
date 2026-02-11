# Use official Apache Spark image with Python support
FROM apache/spark:4.1.0-preview2-scala2.13-java21-python3-r-ubuntu

USER root

# Upgrade pip to support pyproject.toml editable installs
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy and install shared library
COPY shared/ /app/shared/
RUN pip install -e /app/shared[spark]

# Copy and install data-lake service
COPY services/data-lake/ /app/data-lake/
RUN pip install -e /app/data-lake

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Run the streaming job using spark-submit
CMD ["/opt/spark/bin/spark-submit", \
     "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.0-preview2,org.apache.hadoop:hadoop-aws:3.4.2,com.amazonaws:aws-java-sdk-bundle:1.12.792", \
     "/app/data-lake/shrekommender_datalake/spark_jobs/kafka_to_s3.py"]
