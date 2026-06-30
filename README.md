# Fancy LLM Router

A production-grade LLM routing and evaluation system that optimizes for cost, latency, and accuracy across multiple LLM providers.

## Features

- **Dynamic LLM Routing**: Automatically select the best model based on prompt characteristics, cost constraints, and performance benchmarks
- **Comprehensive Metrics**: Track input/output tokens, costs, latency, context size, model parameters, and accuracy
- **Model Drift Detection**: Monitor performance changes across model versions and vendors
- **Prompt Optimization**: Index and refactor prompts for better performance on smaller/cheaper models
- **Session Management**: Support for chained prompts and multi-turn conversations
- **Tool/Skill Integration**: Standardized external resources that work across all models
- **Production Analytics**: Seed evaluation with real production prompts
- **Benchmark Tracking**: Compare against historical performance and manufactured datasets

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Production System                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM Router                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │  Routing    │    │  Metrics    │    │  Session        │  │
│  │  Engine     │◄──►│  Collector  │◄──►│  Manager        │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│        ▲                  ▲                  ▲                │
└────────┼──────────────────┼──────────────────┼────────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────┐  ┌─────────────┐  ┌─────────────────┐
│  Model Providers │  │  Storage     │  │  Tools/Resources │
│  (OpenAI, Anthro,│  │  (SQLite,    │  │  (Mockable       │
│   Local, etc.)   │  │   Postgres)  │  │   External APIs) │
└─────────────────┘  └─────────────┘  └─────────────────┘
```

## Quick Start

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
# Create the virtual environment and install dependencies
uv sync

# Configure your connection and register models interactively
uv run fancy-llm --configure

# ...or copy the example config and edit it by hand
cp configs/example.yaml configs/local.yaml
# Edit configs/local.yaml with your API keys

# Run the evaluation server
uv run python -m fancy_llm_router.api.server

# Or use the CLI
uv run fancy-llm evaluate --prompt "Your prompt here" --models gpt-4,gpt-3.5-turbo,claude-3
```

### Interactive Launcher

For a guided, menu-driven way to run the router, use the [gum](https://github.com/charmbracelet/gum)-powered launcher:

```bash
./run.sh
```

It prompts you to pick a command (`serve`, `route`, `complete`, `chat`, ...) and its options, then runs it via `uv`.

## Development

```bash
# Install dev dependencies (included by default with uv sync)
uv sync

# Run tests
uv run pytest

# Format code
uv run black .

# Lint code
uv run ruff check .

# Type checking
uv run mypy .
```

## Core Concepts

### 1. Model Abstraction
All models implement a standard interface regardless of provider:
- Consistent input/output format
- Token counting
- Cost calculation
- Latency measurement

### 2. Routing Strategies
- **Cost-Optimized**: Choose cheapest model that meets quality thresholds
- **Latency-Optimized**: Choose fastest model that meets quality thresholds
- **Quality-Optimized**: Choose most accurate model within budget
- **Balanced**: Weighted combination of cost, latency, and quality
- **Fallback**: Primary model with automatic fallback on errors

### 3. Metrics Tracking
Every request logs:
- Input/output tokens
- Model parameters and context window
- Cost (input + output tokens × price per token)
- Latency (time to first token, time to complete)
- Quality scores (from evaluators)
- Model version and provider
- Prompt hash and session ID
- Git commit hash (if available)

### 3b. Benchmark & prompt specialization

- **`intent=infer`** (default): normal routing; with `root_id`, substitutes the best
  specialized prompt variant for the chosen deployment.
- **`intent=measure`**: pinned deployment, judge vs `expected_answer`, baseline
  stored in SQLite; optional refactor loop via `PromptOptimizer`.
- **CLI**: `uv run fancy-llm benchmark -r small-01 -p "What is the capital of France?" -e Paris`
- **API**: `POST /api/v1/benchmark/baseline`, `GET /api/v1/analytics/baseline/{run_id}`
- **Dashboard**: `http://localhost:8000/analytics` — parent prompt, tuned variant, response, and telemetry per deployment, with per-root summaries for cost, latency, and tokens
- **Summary API**: `GET /api/v1/analytics/runs`, `GET /api/v1/analytics/baseline/{run_id}/summary`

### 4. Session Management
- Chain multiple prompts together
- Maintain conversation context
- Track cumulative metrics across a session
- Support entry/exit prompts for workflows

### 5. Tools and Skills
- Standardized interface for external resources
- Mock implementations for testing
- Consistent availability across all models
- Versioned tool definitions

## Offline / mock models

For testing the routing + metrics pipeline (or driving it from a load generator)
without any API keys or network access, use the bundled mock config:

```bash
uv run fancy-llm -c configs/mock.yaml serve
# then, e.g.:
curl -s -X POST http://localhost:8000/api/v1/complete \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

The [Simple Token Burner](../simple_token_burner_app) app can drive this server via
its `router` provider:

```bash
uv run main.py --provider router --base-url http://localhost:8000 --max-prompts 10
```

## Configuration

See [configs/](configs/) for example configurations.

### Interactive configuration wizard

Run the wizard to set up an LLM connection (e.g. an OpenAI-compatible endpoint
such as Nebius Token Factory) and register models:

```bash
uv run fancy-llm --configure                 # writes configs/local.yaml
uv run fancy-llm --configure -o configs/nebius.yaml
# equivalent subcommand form:
uv run fancy-llm configure -o configs/nebius.yaml
```

The wizard is built to be low-friction:

1. **Pick a provider** from a built-in list (`nebius`, `openai`, `anthropic`,
   `mistral`, `hyperbolic`, `together`, `groq`, `openrouter`, `deepinfra`,
   `fireworks`, `ollama`, `vllm`, `mock`, `custom`). The **API base URL**,
   **protocol**, and **API-key environment variable** are pre-filled for that
   provider - just press Enter to accept (or override any of them).
2. **Auto-discover models** - for OpenAI-compatible providers the wizard calls
   the standard `GET /v1/models` endpoint and lists everything the provider
   offers, so you don't have to copy/paste model names. Select with `all`,
   index ranges (`1,3,5-9`), or substrings (`qwen, mistral`). Context windows
   (and pricing, when the provider returns it, e.g. OpenRouter) are captured
   automatically.
3. **Optionally add economics** - `/v1/models` rarely returns pricing or
   throughput, so after selecting you can optionally paste a pricing table to
   fill in `$/1M`, `Tok/s`, and quantization. You can also skip this and edit
   the YAML later.

If a provider isn't OpenAI-compatible (or you prefer), you can skip discovery
and **paste a pricing table** directly:

```
Name                  /1M in   /1M out   Tok/s   context   quantization
Qwen/Qwen3-32B        $0.10    $0.30     23      41K       FP8
openai/gpt-oss-120b   $0.15    $0.60     40      131K      FP4
zai-org/GLM-5.2       $1.40    $4.40     54      1,000K    FP8
```

Per-million prices are converted to per-token prices, context sizes like `41K`
or `1,000K` are expanded, and `Tok/s`/`quantization` are stored as first-class
`tokens_per_second` and `quantization` fields on each model. The API key is
stored as a `${ENV_VAR}` reference rather than a literal secret. Each model is
written as a **deployment** keyed `<model>@<source>` (see below), so re-running
the wizard with a different provider registers additional deployments instead of
overwriting existing ones.

### Deployments: one model, many sources

A **deployment** is the routable unit: one logical model served by one source.
The same logical model can be served by several sources at once (Nebius, a local
Ollama box, Hyperbolic, ...), each with its own connection, pricing, throughput,
and quantization. Identity is split across three fields:

| Field           | Meaning                                                              | Example                       |
| --------------- | -------------------------------------------------------------------- | ----------------------------- |
| *mapping key*   | unique **deployment id** (convention `<model>@<source>`)             | `qwen3-32b@nebius`            |
| `model`         | **logical model** a caller asks for (many deployments may share one) | `Qwen/Qwen3-32B`              |
| `model_id`      | **wire id** sent to that host's API (may differ per source)          | `qwen3:32b` (Ollama)          |
| `provider`      | API **protocol** the source speaks                                   | `openai`, `anthropic`, `mock` |
| `source`        | human label for the host                                             | `nebius`, `ollama`            |

```yaml
models:
  qwen3-32b@nebius:
    model: Qwen/Qwen3-32B      # logical name callers request
    source: nebius
    provider: openai           # Nebius speaks the OpenAI protocol
    model_id: Qwen/Qwen3-32B   # wire id sent to Nebius
    api_base_url: https://api.tokenfactory.nebius.com/v1/
    api_key: ${NEBIUS_API_KEY}
    input_token_price: 1.0e-07
    output_token_price: 3.0e-07
    tokens_per_second: 23.0
  qwen3-32b@ollama:
    model: Qwen/Qwen3-32B      # same logical model...
    source: ollama             # ...different source (free, local)
    provider: openai
    model_id: qwen3:32b        # different wire id
    api_base_url: http://localhost:11434/v1/
    input_token_price: 0.0
    output_token_price: 0.0
    tokens_per_second: 15.0
```

Callers ask for a **logical model** (or `auto`) and the router picks the best
deployment for the active strategy. To pin a specific source, reference its
deployment id instead:

```bash
# Let the router choose among all Qwen/Qwen3-32B deployments by strategy
uv run fancy-llm route -p "hi" -m "Qwen/Qwen3-32B" -s cost_optimized   # -> free Ollama
uv run fancy-llm route -p "hi" -m "Qwen/Qwen3-32B" -s latency_optimized # -> faster source

# Pin one deployment explicitly
uv run fancy-llm route -p "hi" -m "qwen3-32b@nebius"
```

## Extending

Add new models by implementing the `BaseModelProvider` interface in `models/`.
Add new tools by implementing the `BaseTool` interface in `tools/`.

## License

Apache 2.0
