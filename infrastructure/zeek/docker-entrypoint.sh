#!/bin/bash
set -euo pipefail

ZEEK_KAFKA_CONF="/opt/zeek/share/zeek/site/zeek_kafka.zeek"

# Patch broker address in Zeek config at runtime
if [ -n "${KAFKA_BROKER:-}" ]; then
    sed -i "s|127.0.0.1:9092|${KAFKA_BROKER}|g" "$ZEEK_KAFKA_CONF"
    echo "Kafka broker set to: ${KAFKA_BROKER}"
fi

# If user passes a custom command (e.g. zeek -C -r /pcaps/file.pcap local)
if [ $# -gt 0 ]; then
    exec "$@"
fi

# Default: live capture on $INTERFACE
echo "Starting Zeek on interface: ${INTERFACE}"
echo "Kafka broker: ${KAFKA_BROKER}"
exec zeek -C -i "${INTERFACE}" local
