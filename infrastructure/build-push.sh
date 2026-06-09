#!/bin/bash
# build-push.sh — Build and push Docker images to GitHub Container Registry
#
#
# Usage:
#   ./build-push.sh              build + push all images
#   ./build-push.sh build        build only (no push)
#   ./build-push.sh push         push only (already built)

set -euo pipefail

# cd to repo root (one level up from infrastructure/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# config
GHCR_USER="${GHCR_USER:-zefx}"
TAG="${TAG:-latest}"

ZEEK_IMAGE="ghcr.io/${GHCR_USER}/hse-thesis-zeek:${TAG}"
CONSUMER_IMAGE="ghcr.io/${GHCR_USER}/hse-thesis-consumer:${TAG}"

# build
build() {
    echo "=== Building Zeek + zeek-kafka image ==="
    echo "    ${ZEEK_IMAGE}"
    docker build -t "${ZEEK_IMAGE}" \
        -f infrastructure/zeek/Dockerfile .

    echo ""
    echo "=== Building Spark consumer image ==="
    echo "    ${CONSUMER_IMAGE}"
    docker build -t "${CONSUMER_IMAGE}" \
        -f infrastructure/consumer/Dockerfile .

    echo ""
    echo "=== Build complete ==="
    docker images | grep "ghcr.io/${GHCR_USER}" || true
}

# push
push() {
    echo "=== Pushing to ghcr.io ==="
    docker push "${ZEEK_IMAGE}"
    docker push "${CONSUMER_IMAGE}"
    echo ""
    echo "=== Push complete ==="
    echo "  ${ZEEK_IMAGE}"
    echo "  ${CONSUMER_IMAGE}"
    echo ""
    echo "Images are PRIVATE by default."
    echo "To make public: GitHub → Packages → Package settings → Change visibility"
}

# main
case "${1:-all}" in
    build)
        build
        ;;
    push)
        push
        ;;
    all)
        build
        push
        ;;
    *)
        echo "Usage: $0 {build|push|all}"
        exit 1
        ;;
esac
