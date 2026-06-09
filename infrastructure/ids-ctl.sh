#!/bin/bash
# ids-ctl.sh — ids helper
#
# Usage:
#   ./ids-ctl.sh status          show all containers
#   ./ids-ctl.sh logs            follow consumer alerts
#   ./ids-ctl.sh restart         restart consumer (keeps Kafka/Zeek)
#   ./ids-ctl.sh reset-topic     delete and recreate zeek-conn topic
#   ./ids-ctl.sh topics          list Kafka topics
#   ./ids-ctl.sh consume-raw     raw Kafka console consumer (debug)
#   ./ids-ctl.sh stop            stop everything
#   ./ids-ctl.sh start           start everything

set -euo pipefail

# resolve path to docker-compose.yml relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

KAFKA_CONTAINER="ids-kafka"
KAFKA_BIN="/opt/kafka/bin"
TOPIC="zeek-conn"
BOOTSTRAP="localhost:9092"

case "${1:-help}" in
    status)
        docker compose ps
        ;;
    logs)
        docker compose logs -f consumer
        ;;
    restart)
        echo "Restarting consumer..."
        docker compose restart consumer
        echo "Done. Use './ids-ctl.sh logs' to watch output."
        ;;
    reset-topic)
        echo "Deleting topic '$TOPIC'..."
        docker exec "$KAFKA_CONTAINER" "$KAFKA_BIN/kafka-topics.sh" \
            --bootstrap-server "$BOOTSTRAP" --delete --topic "$TOPIC" 2>/dev/null || true
        sleep 2
        echo "Topic will be auto-created on next Zeek write."
        echo "Restarting consumer with fresh offsets..."
        docker compose restart consumer
        echo "Done."
        ;;
    topics)
        docker exec "$KAFKA_CONTAINER" "$KAFKA_BIN/kafka-topics.sh" \
            --bootstrap-server "$BOOTSTRAP" --list
        ;;
    consume-raw)
        echo "Raw Kafka consumer (Ctrl+C to stop):"
        docker exec -it "$KAFKA_CONTAINER" "$KAFKA_BIN/kafka-console-consumer.sh" \
            --bootstrap-server "$BOOTSTRAP" --topic "$TOPIC" --max-messages "${2:-5}"
        ;;
    stop)
        docker compose down
        ;;
    start)
        docker compose up -d
        echo "Started. Use './ids-ctl.sh logs' to watch output."
        ;;
    help|*)
        echo "IDS management helper"
        echo ""
        echo "Usage: $0 {status|logs|restart|reset-topic|topics|consume-raw|stop|start}"
        echo ""
        echo "  status       Show container status"
        echo "  logs         Follow consumer alert output"
        echo "  restart      Restart consumer (keeps Kafka + Zeek running)"
        echo "  reset-topic  Delete zeek-conn topic and restart consumer"
        echo "  topics       List Kafka topics"
        echo "  consume-raw  Show raw Kafka messages (default: 5)"
        echo "  stop         Stop all containers"
        echo "  start        Start all containers"
        ;;
esac
