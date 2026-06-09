# Spark consumer — XGBoost temporal ML + Rule engine
#
# Base image: official Spark with Python3 pre-installed (Ubuntu-based).
# Ships Python 3.10+, giving numpy 2.x compatibility — the joblib artifacts
# pickled in the training venv (Python 3.9+ / numpy 2.x) load cleanly.
#
# Build:
#   docker build -t ids-consumer:latest -f infrastructure/consumer/Dockerfile .
#
# Run:
#   docker run --rm --network=host \
#     -e KAFKA_BROKER=127.0.0.1:9092 \
#     ids-consumer:latest

FROM spark:python3

USER root

# ML dependencies (numpy 2.x-compatible versions will be pulled automatically).
RUN pip install --no-cache-dir \
        xgboost scikit-learn joblib pandas numpy

# Pre-download Spark-Kafka connector + transitive deps into Ivy cache inside the image.
# Without this, first container start hangs for minutes while Spark pulls JARs.
# NOTE: scala 2.13 variant of the connector, matched to Spark 4.1.1.
#
# Implementation notes:
#  * os._exit(0) force-kills the JVM gateway after SparkContext.stop() — without
#    it, non-daemon py4j threads keep the process alive and the build step hangs
#    indefinitely (observed: 10h+ wall time, JVM never exits).
#  * `timeout 1200` is a safety net: if Ivy resolution itself stalls, fail fast
#    instead of hanging the whole build.
RUN printf '%s\n' \
        'from pyspark.sql import SparkSession' \
        'SparkSession.builder.appName("warmup").getOrCreate().stop()' \
        'import os; os._exit(0)' \
        > /tmp/warmup.py && \
    timeout 1200 /opt/spark/bin/spark-submit \
        --master "local[1]" \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1 \
        /tmp/warmup.py ; \
    rc=$? ; \
    rm /tmp/warmup.py ; \
    echo "warmup exit=$rc" ; \
    # os._exit(0) is OK; timeout 124 means stall; any other non-zero is real failure.
    if [ $rc -ne 0 ] && [ $rc -ne 124 ]; then exit $rc; fi ; \
    # Promote resolved JARs to /opt/spark/jars/ so Spark picks them up at runtime
    # WITHOUT Ivy resolution. Runtime user is `spark` with HOME=/nonexistent,
    # so any runtime --packages attempt fails writing to /nonexistent/.ivy2.5.2/.
    # Placing jars under /opt/spark/jars/ bypasses Ivy entirely.
    echo "--- promoting Ivy JARs to /opt/spark/jars/ ---" ; \
    find /root -type f -name '*.jar' -path '*ivy*' -exec cp -n {} /opt/spark/jars/ \; ; \
    test -f /opt/spark/jars/spark-sql-kafka-0-10_2.13-4.1.1.jar && \
    echo "connector JAR promoted: OK" || (echo "FATAL: connector JAR missing" && exit 1)

# Consumer code — chmod ensures spark user can read
COPY --chmod=644 infrastructure/consumer/spark_consumer.py /opt/spark/work-dir/
COPY --chmod=644 infrastructure/consumer/rule_engine.py    /opt/spark/work-dir/

# Bake trained models into image (temporal XGBoost — production model,
# validated 7/7 attacks). Only the 3 files the consumer actually loads.
COPY --chmod=644 models/temporal/xgb_baseline.json             /models/temporal/
COPY --chmod=644 models/temporal/label_encoder_xgb.joblib      /models/temporal/
COPY --chmod=644 models/temporal/label_encoder_cats_xgb.joblib /models/temporal/
RUN chmod 755 /models /models/temporal

ENV KAFKA_BROKER=127.0.0.1:9092
ENV KAFKA_TOPIC=zeek-conn
ENV MODEL_DIR=/models/temporal
ENV CHECKPOINT_DIR=/tmp/spark-ids-checkpoint

WORKDIR /opt/spark/work-dir

# Run as the unprivileged `spark` user (default in the official image).
USER spark

# No --packages: Kafka connector + transitive deps are baked into /opt/spark/jars/
# during build. This avoids Ivy resolution at runtime, which would fail because
# the `spark` user has HOME=/nonexistent and cannot write to ~/.ivy2.5.2/.
ENTRYPOINT ["/opt/spark/bin/spark-submit", \
    "--master", "local[*]", \
    "--driver-memory", "8g", \
    "--conf", "spark.driver.maxResultSize=2g", \
    "--conf", "spark.python.worker.memory=2g", \
    "--conf", "spark.sql.shuffle.partitions=4", \
    "/opt/spark/work-dir/spark_consumer.py"]

# Default args — override via docker run or docker-compose command
CMD ["--enable-temporal", "--enable-rules"]
