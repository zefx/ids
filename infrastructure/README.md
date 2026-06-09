# Deployment Guide

Reproducible Docker-based deployment of the IDS prototype.
Three containers (Kafka, Zeek, Spark consumer) on a single Linux server,
traffic capture from any network interface.

## Components


| Service  | Image                                     | Version                       | Description                              |
| -------- | ----------------------------------------- | ----------------------------- | ---------------------------------------- |
| kafka    | `apache/kafka:latest`                     | KRaft mode                    | Message broker, no Zookeeper             |
| zeek     | `ghcr.io/zefx/hse-thesis-zeek:latest`     | Zeek 7.0.4, zeek-kafka v1.2.0 | Packet capture → JSON conn.log → Kafka |
| consumer | `ghcr.io/zefx/hse-thesis-consumer:latest` | Spark 4.1.1, XGBoost          | ML classification + rule-based detection |
| kafka-ui | `provectuslabs/kafka-ui:latest`           | —                            | Web UI for debugging (optional)          |

## Prerequisites

- Linux server (x86_64), Ubuntu 20.04+
- Docker Engine 24+ and Docker Compose v2
- 16 GB RAM minimum (8 GB for Spark driver + Kafka + Zeek overhead); tested on 64 GB
- Network interface with traffic to monitor

### 1. Create `docker-compose.yml`

```yaml
services:
  kafka:
    image: apache/kafka:latest
    container_name: ids-kafka
    restart: unless-stopped
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@localhost:9093
      KAFKA_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://127.0.0.1:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
      KAFKA_NUM_PARTITIONS: 3
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_LOG_RETENTION_HOURS: 24
      KAFKA_LOG_DIRS: /var/lib/kafka/data
    ports:
      - "9092:9092"
    volumes:
      - kafka_data:/var/lib/kafka/data
    healthcheck:
      test: ["CMD-SHELL", "/opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 10
      start_period: 30s

  zeek:
    image: ghcr.io/zefx/hse-thesis-zeek:latest
    container_name: ids-zeek
    restart: unless-stopped
    network_mode: host
    cap_add:
      - NET_RAW
      - NET_ADMIN
    environment:
      INTERFACE: ${INTERFACE:-eth0}
      KAFKA_BROKER: 127.0.0.1:9092
    depends_on:
      kafka:
        condition: service_healthy

  consumer:
    image: ghcr.io/zefx/hse-thesis-consumer:latest
    container_name: ids-consumer
    restart: unless-stopped
    network_mode: host
    environment:
      KAFKA_BROKER: 127.0.0.1:9092
      KAFKA_TOPIC: zeek-conn
      MODEL_DIR: /models/temporal
      CHECKPOINT_DIR: /tmp/spark-ids-checkpoint
    depends_on:
      kafka:
        condition: service_healthy
    command:
      - "--broker"
      - "127.0.0.1:9092"
      - "--topic"
      - "zeek-conn"
      - "--model-dir"
      - "/models/temporal"
      - "--enable-temporal"
      - "--enable-rules"
      - "--trigger"
      - "1 second"

volumes:
  kafka_data:
    driver: local
```

### 2. Create `.env`

```bash
NTERFACE=eth0    #  change to your network interface
```

3. Start

```bash
docker compose up -d
docker compose logs -f consumer
```

Expected startup: Kafka healthy (~30s) → Zeek prints `Starting Zeek on interface: <INTERFACE>` and Consumer prints `Listening for Zeek flows...`

### 4. Test

From a separate Linux machine on the same network:

```bash
# SYN Flood → detected by Rule engine
sudo hping3 -S -p 80 --flood <target-ip>

# HTTP Flood (Hulk) → detected by ML temporal model
# Requires HTTP server on the target — install nginx if needed: sudo apt install -y nginx && systemctl start nginx
# Thesis evaluation used hulk_py2.py (Python 2 fork). Go-port is an alternative if Python 2 is unavailable:
go install github.com/grafov/hulk@latest && ~/go/bin/hulk -site http://<target-ip>

# GoldenEye → detected by ML
git clone https://github.com/jseidl/GoldenEye.git
python3 GoldenEye/goldeneye.py http://<target-ip> -w 50 -s 500

# Slowloris → detected by Rule engine
pip3 install slowloris && slowloris <target-ip> -p 80 -s 500
```

Watch alerts: `docker compose logs -f consumer`
