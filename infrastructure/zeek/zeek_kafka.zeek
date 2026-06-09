# zeek_kafka.zeek — конфигурация zeek-kafka plugin
#
# Публикует conn.log и http.log в отдельные Kafka топики в реальном времени.
# Загружается через local.zeek (@load zeek_kafka)
#
# Топики:
#   zeek-conn  — conn.log  → Consumer A (L3/L4 модель, baseline + temporal)
#   zeek-http  — http.log  → Consumer B (L7 модель, future work)

@load packages

redef Kafka::kafka_conf = table(
    # Kafka broker — Docker на usrv01, INTERNAL listener (port 9092).
    # Zeek runs on host, Kafka in Docker with port mapping 9092:9092.
    # For production: replace with actual broker IP or service discovery.
    ["metadata.broker.list"] = "127.0.0.1:9092",

    # Производительность producer
    ["queue.buffering.max.ms"]       = "100",   # буфер 100ms перед отправкой
    ["queue.buffering.max.messages"] = "100000",
    ["batch.num.messages"]           = "1000",
    ["compression.codec"]            = "snappy", # сжатие — меньше сеть
    ["socket.keepalive.enable"]      = "true"
);

# ── conn.log → topic: zeek-conn ───────────────────────────────────────────────
redef Kafka::topic_name = "zeek-conn";
redef Kafka::logs_to_send = set(Conn::LOG);

# Отправлять только нужные поля conn.log (экономия трафика и памяти Kafka)
# Consumer ожидает именно эти поля
redef Kafka::send_all_active_logs = F;

# ── http.log → topic: zeek-http ───────────────────────────────────────────────
# Kafka::log_policy hook не поддерживается в zeek-kafka v1.2.0
# http.log routing — future work (Consumer B, L7 модель)

# ── Временная метка в Unix timestamp (нужна для temporal features) ────────────
redef LogAscii::use_json = T;          # JSON формат (не TSV)
redef Log::default_rotation_interval = 0sec; # не ротировать — стримим в Kafka

# ── Cluster mode: только workers публикуют в Kafka ───────────────────────────
# Manager и logger не захватывают пакеты → не должны публиковать
@if ( Cluster::is_enabled() )
    @if ( Cluster::local_node_type() != Cluster::WORKER )
        redef Kafka::logs_to_send = set();  # выключить на non-worker нодах
    @endif
@endif
