#!/usr/bin/env bash
#
# Interactive launcher for the Fancy LLM Router.
# Uses `gum` to prompt for a command and its options, then runs via `uv run`.

set -euo pipefail

cd "$(dirname "$0")"

# Load API keys and other secrets from .env (e.g. NEBIUS_API_KEY for configs/local.yaml).
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

usage() {
    cat <<'EOF'
Fancy LLM Router - interactive launcher

USAGE:
    ./run.sh [-h|--help]

DESCRIPTION:
    A gum-powered, menu-driven launcher that prompts you for a command and its
    options, then runs it via `uv run fancy-llm`.

    Running with no arguments starts the interactive menu, which supports:
      - configure     Set up an LLM connection and register models
      - serve         Start the API server
      - list-models   List available models (optionally filtered)
      - route         Test routing without executing
      - complete      Generate a text completion
      - chat          Generate a chat completion
      - evaluate      Run evaluation benchmarks

    Command-specific prompts collect the prompt/messages, routing strategy,
    model overrides, token limits, and JSON output options.

OPTIONS:
    -h, --help    Show this help message and exit.

REQUIREMENTS:
    gum   https://github.com/charmbracelet/gum   (brew install gum)
    uv    https://docs.astral.sh/uv/             (brew install uv)

ENVIRONMENT:
    If a .env file exists in the project root, it is sourced before running
    (e.g. NEBIUS_API_KEY for Nebius Token Factory models in configs/local.yaml).
EOF
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
esac

if ! command -v gum >/dev/null 2>&1; then
    echo "Error: 'gum' is not installed."
    echo "Install it with: brew install gum"
    echo "See https://github.com/charmbracelet/gum for other platforms."
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: 'uv' is not installed."
    echo "Install it with: brew install uv"
    echo "See https://docs.astral.sh/uv/ for other platforms."
    exit 1
fi

STRATEGIES="balanced cost_optimized latency_optimized quality_optimized fallback random round_robin custom"

gum style --border normal --margin "1" --padding "1 2" --border-foreground 212 \
    "Fancy LLM Router"

COMMAND=$(gum choose --header "Select a command" \
    "configure" \
    "serve" \
    "list-models" \
    "route" \
    "complete" \
    "chat" \
    "evaluate")

cmd=(uv run fancy-llm)

# if gum confirm "Enable verbose logging?" --default=no; then
#     cmd+=(--verbose)
# fi

cmd+=("$COMMAND")

case "$COMMAND" in
    "complete")
        PROMPT=$(gum input --header "Prompt" --placeholder "Explain quantum computing")
        if [ -z "$PROMPT" ]; then
            echo "A prompt is required. Aborting."
            exit 1
        fi
        STRATEGY=$(gum choose --header "Routing strategy" $STRATEGIES)
        MODEL=$(gum input --header "Force a specific model (leave blank to auto-route)" --placeholder "gpt-4")
        MAX_TOKENS=$(gum input --header "Max tokens" --value "256")
        cmd+=("$PROMPT" --strategy "$STRATEGY" --max-tokens "$MAX_TOKENS")
        [ -n "$MODEL" ] && cmd+=(--model "$MODEL")
        if gum confirm "Output as JSON?" --default=no; then
            cmd+=(--json)
        fi
        ;;

    "chat")
        MESSAGES=$(gum write --header "Enter one message per line")
        if [ -z "$MESSAGES" ]; then
            echo "At least one message is required. Aborting."
            exit 1
        fi
        STRATEGY=$(gum choose --header "Routing strategy" $STRATEGIES)
        MODEL=$(gum input --header "Force a specific model (leave blank to auto-route)" --placeholder "gpt-4")
        cmd+=(--strategy "$STRATEGY")
        [ -n "$MODEL" ] && cmd+=(--model "$MODEL")
        if gum confirm "Output as JSON?" --default=no; then
            cmd+=(--json)
        fi
        while IFS= read -r line; do
            [ -n "$line" ] && cmd+=("$line")
        done <<< "$MESSAGES"
        ;;

    "route")
        PROMPT=$(gum input --header "Test prompt (leave blank for default)" --placeholder "Test prompt for routing")
        STRATEGY=$(gum choose --header "Routing strategy" $STRATEGIES)
        cmd+=(--strategy "$STRATEGY")
        [ -n "$PROMPT" ] && cmd+=(--prompt "$PROMPT")
        if gum confirm "Output as JSON?" --default=no; then
            cmd+=(--json)
        fi
        ;;

    "list-models")
        # MODELS=$(gum input --header "Filter to specific models, comma-separated (leave blank for all)" --placeholder "gpt-4,gpt-3.5-turbo")
        # if [ -n "$MODELS" ]; then
        #     IFS=',' read -ra ITEMS <<< "$MODELS"
        #     for m in "${ITEMS[@]}"; do
        #         m=$(echo "$m" | xargs)
        #         [ -n "$m" ] && cmd+=(--models "$m")
        #     done
        # fi
        ;;

    "configure")
        OUTPUT=$(gum input --header "Output config path" --value "configs/local.yaml")
        [ -n "$OUTPUT" ] && cmd+=(--output "$OUTPUT")
        ;;

    "serve"|"evaluate")
        ;;
esac

echo
gum style --foreground 212 "Running: ${cmd[*]}"
echo

exec "${cmd[@]}"
