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

```bash
# Install dependencies
pip install -e .

# Configure your models
cp configs/example.yaml configs/local.yaml
# Edit configs/local.yaml with your API keys

# Run the evaluation server
python -m fancy_llm_router.api.server

# Or use the CLI
fancy-llm evaluate --prompt "Your prompt here" --models gpt-4,gpt-3.5-turbo,claude-3
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

## Configuration

See [configs/](configs/) for example configurations.

## Extending

Add new models by implementing the `BaseModelProvider` interface in `models/`.
Add new tools by implementing the `BaseTool` interface in `tools/`.

## License

Apache 2.0
