# Zeek 7.0 + zeek-kafka plugin
#
# Zeek needs host network access for promiscuous packet capture.
# Run with: --network=host --cap-add=NET_RAW --cap-add=NET_ADMIN
#
# Build:
#   docker build -t ghcr.io/zefx/hse-thesis-zeek:latest -f infrastructure/zeek/Dockerfile .
#
# Run (live capture):
#   docker run --rm --network=host \
#     --cap-add=NET_RAW --cap-add=NET_ADMIN \
#     -e INTERFACE=enp2s0 \
#     -e KAFKA_BROKER=127.0.0.1:9092 \
#     zeek-kafka:latest
#
# Run (pcap replay):
#   docker run --rm -v /path/to/pcaps:/pcaps \
#     -e KAFKA_BROKER=127.0.0.1:9092 \
#     zeek-kafka:latest \
#     zeek -C -r /pcaps/capture.pcap local

FROM zeek/zeek:7.0.4

LABEL maintainer="HSE Thesis 2026"
LABEL description="Zeek with zeek-kafka plugin for real-time conn.log streaming"

# System deps for zeek-kafka plugin build
# - librdkafka-dev: Kafka client library
# - libpcap-dev: required by Zeek headers (Packet.h includes pcap.h for DLT_ constants)
# - libssl-dev, libsasl2-dev: TLS/SASL support in librdkafka
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        git cmake make g++ \
        librdkafka-dev libpcap-dev libssl-dev libsasl2-dev && \
    rm -rf /var/lib/apt/lists/*

# Git configuration — larger buffers to survive flaky TLS during cross-platform build
RUN git config --global http.postBuffer 524288000 && \
    git config --global http.lowSpeedLimit 0 && \
    git config --global http.lowSpeedTime 999999

# Ensure /opt/zeek symlink exists (some tools expect it, base image may use /usr/local/zeek)
RUN ZEEK_PREFIX=$(zeek-config --prefix) && \
    echo "Zeek prefix: $ZEEK_PREFIX" && \
    [ "$ZEEK_PREFIX" = "/opt/zeek" ] || ln -sf "$ZEEK_PREFIX" /opt/zeek

# Install zeek-kafka via zkg.
# zeek-kafka needs a Zeek source tree for manual ./configure, which we don't have
# (the base image ships only the binary install). zkg uses zeek-config to find
# include/cmake dirs of the binary install, which is the only way that works here.
#
# On failure, zkg hides the real compiler error — so we re-run make manually
# in its clone directory with -j1 VERBOSE=1 to expose the actual g++ output.
ARG ZEEK_KAFKA_VERSION=v1.2.0
RUN zkg autoconfig --force && \
    (zkg install --force --skiptests \
        --version ${ZEEK_KAFKA_VERSION} \
        https://github.com/SeisoLLC/zeek-kafka 2>&1 \
     || (echo "===== zkg install FAILED — dumping zkg's saved build log =====" && \
         echo "--- searching for zkg logs ---" && \
         find / -name "zeek-kafka-build.log" 2>/dev/null | while read f; do \
             echo "===== $f ====="; cat "$f"; \
         done ; \
         echo "--- searching for zkg clone dirs ---" && \
         find / -type d -name "zeek-kafka" 2>/dev/null | while read d; do \
             echo "===== clone: $d ====="; \
             ls -la "$d" 2>/dev/null ; \
             BUILD="$d/build"; \
             if [ -d "$BUILD" ]; then \
                 echo "--- re-running make in $BUILD ---"; \
                 (cd "$BUILD" && make -j1 VERBOSE=1 2>&1 | tail -150) ; \
             fi ; \
         done ; \
         exit 1))

# Register plugin so Zeek loads it
RUN echo '@load-plugin Seiso::Kafka' >> $(zeek-config --site_dir)/local.zeek

# Verify plugin loads
RUN zeek -NN Seiso::Kafka

# Copy Zeek config (will be patched at runtime with KAFKA_BROKER)
COPY infrastructure/zeek/zeek_kafka.zeek /opt/zeek/share/zeek/site/zeek_kafka.zeek

# Add @load to local.zeek
RUN echo '@load zeek_kafka' >> /opt/zeek/share/zeek/site/local.zeek

# Entrypoint script — patches broker IP and starts Zeek
COPY infrastructure/zeek/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENV INTERFACE=eth0
ENV KAFKA_BROKER=127.0.0.1:9092

ENTRYPOINT ["/docker-entrypoint.sh"]
