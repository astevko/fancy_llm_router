#!/usr/bin/env bash
#
# Remove the Fancy LLM Router SQLite database (metrics + baseline analytics).
# Tables are recreated automatically the next time the server starts.

set -euo pipefail

cd "$(dirname "$0")"

CONFIG=""
DB_PATH=""
FORCE=0

usage() {
    cat <<'EOF'
Wipe Fancy LLM Router SQLite database

USAGE:
    ./wipe-db.sh [-h|--help]
    ./wipe-db.sh [--force] [--config PATH] [--db PATH]

OPTIONS:
    -h, --help        Show this help message
    -f, --force       Delete without confirmation
    -c, --config PATH Read storage.sqlite_path from a router YAML config
    --db PATH         Explicit database file (default: data/metrics.db)

EXAMPLES:
    ./wipe-db.sh
    ./wipe-db.sh --force
    ./wipe-db.sh --config configs/local.yaml
    ./wipe-db.sh --db data/metrics.db --force

NOTE:
    Stop the router server before wiping, or you may delete a file that is
    still open. After wiping, restart with:
      uv run fancy-llm -c configs/local.yaml serve
EOF
}

resolve_db_from_config() {
    local config_path="$1"
    if [[ ! -f "$config_path" ]]; then
        echo "Error: config not found: $config_path" >&2
        exit 1
    fi
    uv run python - <<'PY' "$config_path"
import sys
from fancy_llm_router.core.config_loader import get_storage_db_path, load_config

path = sys.argv[1]
print(get_storage_db_path(load_config(path)))
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -f|--force)
            FORCE=1
            shift
            ;;
        -c|--config)
            CONFIG="${2:-}"
            if [[ -z "$CONFIG" ]]; then
                echo "Error: --config requires a path" >&2
                exit 1
            fi
            shift 2
            ;;
        --db)
            DB_PATH="${2:-}"
            if [[ -z "$DB_PATH" ]]; then
                echo "Error: --db requires a path" >&2
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -n "$CONFIG" ]]; then
    DB_PATH="$(resolve_db_from_config "$CONFIG")"
elif [[ -z "$DB_PATH" ]]; then
    DB_PATH="data/metrics.db"
fi

# Normalize relative paths against project root.
if [[ "$DB_PATH" != /* ]]; then
    DB_PATH="$(pwd)/$DB_PATH"
fi

if [[ ! -f "$DB_PATH" ]]; then
    echo "Nothing to wipe: $DB_PATH does not exist."
    exit 0
fi

SIZE="$(wc -c <"$DB_PATH" | tr -d ' ')"
echo "Database: $DB_PATH"
echo "Size:     $SIZE bytes"

if [[ "$FORCE" -ne 1 ]]; then
    read -r -p "Delete this database? [y/N] " answer
    case "${answer:-}" in
        y|Y|yes|YES) ;;
        *)
            echo "Aborted."
            exit 0
            ;;
    esac
fi

rm -f "$DB_PATH"
# SQLite WAL/SHM sidecars, if present
rm -f "${DB_PATH}-wal" "${DB_PATH}-shm"

echo "Deleted $DB_PATH"
echo "Restart the router to recreate an empty database."
