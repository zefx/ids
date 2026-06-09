#!/bin/bash
# process_cicids.sh
# Прогоняет все pcap (DCAP + UCAP) через Zeek, собирает conn.log и http.log
# по дням и в объединённый файл.
#
# Если Zeek не может прочитать файл (truncated dump) — автоматически
# исправляет через editcap и повторяет попытку.
#
# Структура output:
#   /data/cicids_zeek/
#     Wednesday-14-02-2018/
#       conn.log      ← все pcap этого дня (DCAP + UCAP)
#       http.log
#     ...
#     combined/
#       conn.log      ← все дни, отсортировано по timestamp
#       http.log
#
# Usage:
#   bash process_cicids.sh           # пропускает уже готовые дни
#   bash process_cicids.sh --force   # перепрогнать все дни заново

set -euo pipefail

# ── CONFIGURE ────────────────────────────────────────────────────────────────
NAS_BASE="/mnt/nas/cicids/CICIDS/CICIDS18/Original Network Traffic and Log data"
OUTPUT_BASE="/data/cicids_zeek"
COMBINED_DIR="$OUTPUT_BASE/combined"
ZEEK_BIN="zeek"
LOG="$OUTPUT_BASE/process.log"
FORCE=false
# ─────────────────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
    echo ">>> --force: все дни будут перепрогнаны"
fi

which editcap > /dev/null 2>&1 || {
    echo "editcap не найден. Установить: sudo apt install wireshark-common"
    exit 1
}

mkdir -p "$OUTPUT_BASE" "$COMBINED_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "========================================="
echo "CICIDS2018 Zeek processing started: $(date)"
echo "Force reprocess: $FORCE"
echo "========================================="

rm -f "$COMBINED_DIR/conn.log" "$COMBINED_DIR/http.log"
combined_conn_header=false
combined_http_header=false

for day_dir in "$NAS_BASE"/*/; do
    day_name=$(basename "$day_dir")
    pcap_dir="$day_dir/pcap"

    if [ ! -d "$pcap_dir" ]; then
        echo "[SKIP] $day_name — no pcap/ subdir"
        continue
    fi

    day_out="$OUTPUT_BASE/$day_name"
    day_conn="$day_out/conn.log"
    day_http="$day_out/http.log"

    # Пропустить уже готовые (если не --force)
    if [ "$FORCE" = false ] && [ -f "$day_conn" ]; then
        echo "[SKIP] $day_name — already processed (use --force to redo)"
        if [ "$combined_conn_header" = false ]; then
            cat "$day_conn" >> "$COMBINED_DIR/conn.log"
            combined_conn_header=true
        else
            grep -v "^#" "$day_conn" >> "$COMBINED_DIR/conn.log" || true
        fi
        [ -f "$day_http" ] && {
            if [ "$combined_http_header" = false ]; then
                cat "$day_http" >> "$COMBINED_DIR/http.log"
                combined_http_header=true
            else
                grep -v "^#" "$day_http" >> "$COMBINED_DIR/http.log" || true
            fi
        }
        continue
    fi

    mkdir -p "$day_out"
    # Очистить старые логи при перепрогоне
    rm -f "$day_conn" "$day_http"

    echo ""
    echo "--- $day_name  [$(date +%H:%M:%S)] ---"

    day_conn_header=false
    day_http_header=false
    pcap_count=0
    skip_count=0
    editcap_count=0

    for pcap_file in "$pcap_dir"/*; do
        [ -f "$pcap_file" ] || continue
        cap_name=$(basename "$pcap_file")
        pcap_count=$((pcap_count + 1))

        tmp_dir=$(mktemp -d /tmp/zeek_XXXXXX)
        input_file="$pcap_file"
        tmp_fixed=""

        echo "  [zeek] $cap_name"

        # Попытка 1: запустить Zeek напрямую
        if ! (cd "$tmp_dir" && "$ZEEK_BIN" -C -r "$input_file" 2>/dev/null); then

            # Попытка 2: исправить через editcap и повторить
            tmp_fixed=$(mktemp /tmp/zeek_fixed_XXXXXX.pcap)
            editcap "$input_file" "$tmp_fixed" 2>/dev/null || true

            if [ -s "$tmp_fixed" ]; then
                editcap_count=$((editcap_count + 1))
                echo "  [editcap+retry] $cap_name"
                if ! (cd "$tmp_dir" && "$ZEEK_BIN" -C -r "$tmp_fixed" 2>/dev/null); then
                    echo "  [WARN] failed even after editcap: $cap_name"
                    skip_count=$((skip_count + 1))
                    rm -f "$tmp_fixed"
                    rm -rf "$tmp_dir"
                    continue
                fi
            else
                echo "  [WARN] editcap produced empty file: $cap_name"
                skip_count=$((skip_count + 1))
                rm -f "$tmp_fixed"
                rm -rf "$tmp_dir"
                continue
            fi
        fi

        # Собрать conn.log
        if [ -f "$tmp_dir/conn.log" ]; then
            if [ "$day_conn_header" = false ]; then
                cat "$tmp_dir/conn.log" >> "$day_conn"
                day_conn_header=true
            else
                grep -v "^#" "$tmp_dir/conn.log" >> "$day_conn" || true
            fi
        fi

        # Собрать http.log
        if [ -f "$tmp_dir/http.log" ]; then
            if [ "$day_http_header" = false ]; then
                cat "$tmp_dir/http.log" >> "$day_http"
                day_http_header=true
            else
                grep -v "^#" "$tmp_dir/http.log" >> "$day_http" || true
            fi
        fi

        [ -n "$tmp_fixed" ] && rm -f "$tmp_fixed"
        rm -rf "$tmp_dir"
    done

    echo "  Done: $pcap_count files, $editcap_count fixed via editcap, $skip_count failed"

    if [ ! -f "$day_conn" ]; then
        echo "  [WARN] no conn.log for $day_name"
        continue
    fi

    # Сортировка по timestamp
    echo "  Sorting..."
    sorted_tmp="$day_out/conn_sorted.tmp"
    grep "^#" "$day_conn" > "$sorted_tmp"
    grep -v "^#" "$day_conn" | sort -k1,1 -n >> "$sorted_tmp"
    mv "$sorted_tmp" "$day_conn"

    conn_rows=$(grep -vc "^#" "$day_conn" 2>/dev/null || echo 0)
    http_rows=0
    [ -f "$day_http" ] && http_rows=$(grep -vc "^#" "$day_http" 2>/dev/null || echo 0)
    echo "  conn: $conn_rows rows | http: $http_rows rows"

    # Добавить в combined
    if [ "$combined_conn_header" = false ]; then
        cat "$day_conn" >> "$COMBINED_DIR/conn.log"
        combined_conn_header=true
    else
        grep -v "^#" "$day_conn" >> "$COMBINED_DIR/conn.log" || true
    fi

    [ -f "$day_http" ] && {
        if [ "$combined_http_header" = false ]; then
            cat "$day_http" >> "$COMBINED_DIR/http.log"
            combined_http_header=true
        else
            grep -v "^#" "$day_http" >> "$COMBINED_DIR/http.log" || true
        fi
    }
done

# Финальная сортировка combined
echo ""
echo "--- Sorting combined conn.log ---"
sorted_tmp="$COMBINED_DIR/conn_sorted.tmp"
grep "^#" "$COMBINED_DIR/conn.log" > "$sorted_tmp"
grep -v "^#" "$COMBINED_DIR/conn.log" | sort -k1,1 -n >> "$sorted_tmp"
mv "$sorted_tmp" "$COMBINED_DIR/conn.log"

total=$(grep -vc "^#" "$COMBINED_DIR/conn.log" 2>/dev/null || echo 0)

echo ""
echo "========================================="
echo "Zeek done: $(date)"
echo "Combined conn.log: $total rows"
echo "Next step: bash label_all.sh"
echo "========================================="
