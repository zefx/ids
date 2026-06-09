#!/bin/bash
# label_all.sh
# Шаг 2: запускает label_zeek_v2.py на каждом дне и на combined.
# Запускать после того как process_cicids.sh завершился.
#
# Excluded days (all attacks are application-layer — see EXCLUDED_ATTACKS in label_zeek_v2.py):
#   Thursday-22-02-2018  — Brute Force -Web, XSS, SQL Injection
#   Friday-23-02-2018    — Brute Force -Web, XSS, SQL Injection
#   Wednesday-28-02-2018 — Infiltration
#   Thursday-01-03-2018  — Infiltration
#
# Usage: bash label_all.sh

set -euo pipefail

OUTPUT_BASE="/data/cicids_zeek"
COMBINED_DIR="$OUTPUT_BASE/combined"
LABEL_SCRIPT="$HOME/label_zeek_v2.py"
SCHEDULE="$HOME/cse_cic_ids2018_attack_schedule.xlsx"

# Только дни с сетевыми атаками (L3/L4) — L7-дни исключены
declare -A DAY_MAP=(
    ["Wednesday-14-02-2018"]="2018-02-14"   # FTP-BruteForce, SSH-Bruteforce
    ["Thursday-15-02-2018"]="2018-02-15"    # DoS-GoldenEye, DoS-Slowloris
    ["Friday-16-02-2018"]="2018-02-16"      # DoS-Hulk, DoS-SlowHTTPTest
    ["Wednesday-21-02-2018"]="2018-02-21"   # DDOS-HOIC, DDOS-LOIC-UDP
    ["Friday-02-03-2018"]="2018-03-02"      # Bot
)

echo "========================================="
echo "Labeling started: $(date)"
echo "========================================="

# Разметка каждого дня
for DIR in "${!DAY_MAP[@]}"; do
    DAY="${DAY_MAP[$DIR]}"
    CONN="$OUTPUT_BASE/$DIR/conn.log"
    OUT="$OUTPUT_BASE/$DIR/labeled.csv"

    if [ ! -f "$CONN" ]; then
        echo "[skip] $DIR — no conn.log"
        continue
    fi

    echo "--- $DIR ($DAY) ---"
    python3 "$LABEL_SCRIPT" \
        --conn "$CONN" \
        --schedule "$SCHEDULE" \
        --day "$DAY" \
        --out "$OUT" 2>&1 | grep -E "Total|Label|BENIGN|DoS|Bot|Brute|SQL|DDoS|Infiltration|SSH|FTP|\[3"
done

# Combined — склеить per-day labeled.csv (намного быстрее чем re-label 56M строк)
echo ""
echo "--- Merging per-day labeled.csv into combined ---"
COMBINED_CSV="$COMBINED_DIR/labeled.csv"
rm -f "$COMBINED_CSV"
first=true
for DIR in "${!DAY_MAP[@]}"; do
    CSV="$OUTPUT_BASE/$DIR/labeled.csv"
    [ -f "$CSV" ] || continue
    if [ "$first" = true ]; then
        cat "$CSV" >> "$COMBINED_CSV"
        first=false
    else
        tail -n +2 "$CSV" >> "$COMBINED_CSV"
    fi
done
echo "Combined labeled.csv: $(wc -l < "$COMBINED_CSV") lines"

echo ""
echo "========================================="
echo "Summary:"
for DIR in "${!DAY_MAP[@]}"; do
    CSV="$OUTPUT_BASE/$DIR/labeled.csv"
    [ -f "$CSV" ] || continue
    echo "--- $DIR ---"
    tail -n +2 "$CSV" | awk -F',' '{print $NF}' | sort | uniq -c | sort -rn
done

echo ""
echo "--- COMBINED ---"
if [ -f "$COMBINED_CSV" ]; then
    tail -n +2 "$COMBINED_CSV" | awk -F',' '{print $NF}' | sort | uniq -c | sort -rn
fi

echo "========================================="
echo "Done: $(date)"
