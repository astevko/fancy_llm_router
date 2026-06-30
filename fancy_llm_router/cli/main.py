"""Main CLI for the LLM Router."""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

from fancy_llm_router.core.router import LLMRouter, RoutingStrategy
from fancy_llm_router.core.config_loader import create_router, find_default_config
from fancy_llm_router.metrics.collector import MetricsCollector
from fancy_llm_router.schemas.requests import CompletionRequest, ChatRequest
from fancy_llm_router.schemas.routing import RoutingCriteria

console = Console()
logger = logging.getLogger(__name__)


def _get_router(ctx: Optional[click.Context]) -> LLMRouter:
    """Build a router from the config path stored on the CLI context."""
    config_path = None
    if ctx is not None and ctx.obj:
        config_path = ctx.obj.get("config_path")
    try:
        return create_router(config_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Failed to load configuration:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration wizard helpers
# ---------------------------------------------------------------------------

def _parse_price_per_million(value: str) -> float:
    """Parse a '/1M' price like '$0.10' into a per-token price in USD.

    Uses ``Decimal`` for the division so the stored per-token value is a clean
    number (e.g. ``1e-07``) instead of a binary-float artifact such as
    ``1.0000000000000001e-07``.
    """
    cleaned = value.strip().lstrip("$").replace(",", "").strip()
    if not cleaned:
        raise ValueError("empty price")
    try:
        return float(Decimal(cleaned) / Decimal(1_000_000))
    except InvalidOperation as exc:
        raise ValueError(f"invalid price: {value!r}") from exc


def _slug(value: str) -> str:
    """Make a filesystem/key-friendly slug from a model or source name."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "model"


def _parse_context_window(value: str) -> int:
    """Parse a context size like '41K', '1,000K', or '131072' into an int."""
    cleaned = value.strip().replace(",", "").upper()
    if not cleaned:
        raise ValueError("empty context window")
    multiplier = 1
    if cleaned.endswith("K"):
        multiplier = 1_000
        cleaned = cleaned[:-1]
    elif cleaned.endswith("M"):
        multiplier = 1_000_000
        cleaned = cleaned[:-1]
    return int(round(float(cleaned) * multiplier))


def _parse_model_table(text: str) -> List[Dict[str, Any]]:
    """Parse a pasted pricing table into model definitions.

    Expected columns (whitespace/tab separated), with the model name allowed to
    contain spaces:

        Name   /1M in   /1M out   Tok/s   context   quantization
    """
    models: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 6:
            continue
        # The five right-most columns are fixed; everything before is the name.
        in_price, out_price, toks, context, quant = tokens[-5:]
        name = " ".join(tokens[:-5])
        try:
            model = {
                "model_id": name,
                "input_token_price": _parse_price_per_million(in_price),
                "output_token_price": _parse_price_per_million(out_price),
                "input_price_per_1m": float(in_price.lstrip("$").replace(",", "")),
                "output_price_per_1m": float(out_price.lstrip("$").replace(",", "")),
                "tokens_per_second": float(toks.replace(",", "")),
                "context_window": _parse_context_window(context),
                "quantization": quant,
            }
        except (ValueError, ArithmeticError):
            # Header rows or malformed lines are silently skipped.
            continue
        models.append(model)
    return models


def _collect_model_table() -> List[Dict[str, Any]]:
    """Collect a pricing table from the user (editor or stdin) and parse it."""
    console.print(
        "\n[bold]Paste your model pricing table.[/bold] Columns:\n"
        "  [cyan]Name  /1M in  /1M out  Tok/s  context  quantization[/cyan]\n"
        "Example: [dim]Qwen/Qwen3-32B  $0.10  $0.30  23  41K  FP8[/dim]\n"
    )

    text: Optional[str] = None
    if Confirm.ask("Open an editor to paste the table?", default=True):
        text = click.edit(
            "# Paste rows below this line, then save and close.\n"
            "# Format: Name  /1M in  /1M out  Tok/s  context  quantization\n"
        )

    if not text:
        console.print("Paste rows now. Enter a line with [bold]END[/bold] to finish:")
        lines: List[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "END":
                break
            lines.append(line)
        text = "\n".join(lines)

    return _parse_model_table(text or "")


# ---------------------------------------------------------------------------
# Known providers: default base URL + protocol + API-key env var per source.
# `openai_compatible` marks endpoints that expose the standard `GET /v1/models`
# listing so the wizard can auto-discover models without copy/paste.
# ---------------------------------------------------------------------------

KNOWN_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "nebius": {
        "protocol": "openai",
        "base_url": "https://api.tokenfactory.nebius.com/v1/",
        "env": "NEBIUS_API_KEY",
        "openai_compatible": True,
    },
    "openai": {
        "protocol": "openai",
        "base_url": "https://api.openai.com/v1/",
        "env": "OPENAI_API_KEY",
        "openai_compatible": True,
    },
    "anthropic": {
        "protocol": "anthropic",
        "base_url": "https://api.anthropic.com/v1/",
        "env": "ANTHROPIC_API_KEY",
        "openai_compatible": False,
    },
    "mistral": {
        "protocol": "openai",
        "base_url": "https://api.mistral.ai/v1/",
        "env": "MISTRAL_API_KEY",
        "openai_compatible": True,
    },
    "hyperbolic": {
        "protocol": "openai",
        "base_url": "https://api.hyperbolic.xyz/v1/",
        "env": "HYPERBOLIC_API_KEY",
        "openai_compatible": True,
    },
    "together": {
        "protocol": "openai",
        "base_url": "https://api.together.xyz/v1/",
        "env": "TOGETHER_API_KEY",
        "openai_compatible": True,
    },
    "groq": {
        "protocol": "openai",
        "base_url": "https://api.groq.com/openai/v1/",
        "env": "GROQ_API_KEY",
        "openai_compatible": True,
    },
    "openrouter": {
        "protocol": "openai",
        "base_url": "https://openrouter.ai/api/v1/",
        "env": "OPENROUTER_API_KEY",
        "openai_compatible": True,
    },
    "deepinfra": {
        "protocol": "openai",
        "base_url": "https://api.deepinfra.com/v1/openai/",
        "env": "DEEPINFRA_API_KEY",
        "openai_compatible": True,
    },
    "fireworks": {
        "protocol": "openai",
        "base_url": "https://api.fireworks.ai/inference/v1/",
        "env": "FIREWORKS_API_KEY",
        "openai_compatible": True,
    },
    "ollama": {
        "protocol": "openai",
        "base_url": "http://localhost:11434/v1/",
        "env": "",
        "openai_compatible": True,
    },
    "vllm": {
        "protocol": "openai",
        "base_url": "http://localhost:8000/v1/",
        "env": "",
        "openai_compatible": True,
    },
    "mock": {
        "protocol": "mock",
        "base_url": "",
        "env": "",
        "openai_compatible": False,
    },
    "custom": {
        "protocol": "openai",
        "base_url": "",
        "env": "",
        "openai_compatible": True,
    },
}

# Protocol values the router can actually instantiate / route with.
PROTOCOLS = [
    "openai", "anthropic", "google", "cohere", "mistral",
    "local", "huggingface", "ollama", "vllm", "custom", "mock",
]


def _discover_models(
    base_url: str,
    api_key: Optional[str] = None,
    timeout: float = 20.0,
) -> List[Dict[str, Any]]:
    """List models from an OpenAI-compatible ``GET {base_url}/models`` endpoint.

    Returns normalized dicts compatible with the pricing-table parser. Pricing,
    throughput, and quantization are usually absent from a bare ``/models``
    listing, so those keys are only included when the provider supplies them
    (e.g. OpenRouter-style ``pricing``); otherwise they default later.
    """
    import httpx

    url = base_url.rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = httpx.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = payload.get("data") or payload.get("models") or []
    else:
        raw = []

    models: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            models.append({"model_id": item})
            continue
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("name") or item.get("model")
        if not model_id:
            continue
        entry: Dict[str, Any] = {"model_id": model_id}

        ctx = (
            item.get("context_length")
            or item.get("context_window")
            or item.get("max_model_len")
            or (item.get("config") or {}).get("context_length")
        )
        if ctx:
            try:
                entry["context_window"] = int(ctx)
            except (TypeError, ValueError):
                pass

        # OpenRouter-style pricing: USD per token, as strings.
        pricing = item.get("pricing") or {}
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")
        try:
            if prompt_price is not None:
                entry["input_token_price"] = float(prompt_price)
                entry["input_price_per_1m"] = round(float(prompt_price) * 1_000_000, 6)
            if completion_price is not None:
                entry["output_token_price"] = float(completion_price)
                entry["output_price_per_1m"] = round(float(completion_price) * 1_000_000, 6)
        except (TypeError, ValueError):
            pass

        models.append(entry)

    models.sort(key=lambda m: m["model_id"].lower())
    return models


def _apply_selection(models: List[Dict[str, Any]], selection: str) -> List[Dict[str, Any]]:
    """Resolve a selection string into a subset of ``models``.

    Accepts ``all``, numeric indices/ranges (``1,3,5-9``), or substring filters
    (``qwen, mistral``) matched case-insensitively against the model id.
    """
    selection = (selection or "").strip()
    if not selection or selection.lower() == "all":
        return list(models)

    if all(ch.isdigit() or ch in ", -" for ch in selection):
        indices: set = set()
        for part in selection.split(","):
            part = part.strip()
            if "-" in part:
                lo, _, hi = part.partition("-")
                try:
                    indices.update(range(int(lo), int(hi) + 1))
                except ValueError:
                    continue
            elif part.isdigit():
                indices.add(int(part))
        return [models[i - 1] for i in sorted(indices) if 1 <= i <= len(models)]

    terms = [t.strip().lower() for t in selection.split(",") if t.strip()]
    return [m for m in models if any(t in m["model_id"].lower() for t in terms)]


def _select_discovered_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Show discovered models and let the user pick which to register."""
    if not models:
        return []

    table = Table(title=f"{len(models)} models offered by the provider")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Model", style="cyan")
    table.add_column("Context", justify="right")
    table.add_column("$/1M in", justify="right")
    for i, m in enumerate(models, 1):
        ctx = m.get("context_window")
        in1m = m.get("input_price_per_1m")
        table.add_row(
            str(i),
            m["model_id"],
            f"{ctx:,}" if ctx else "-",
            f"${in1m:.2f}" if in1m is not None else "-",
        )
    console.print(table)
    console.print(
        "[dim]Select with 'all', indices/ranges like '1,3,5-9', "
        "or substrings like 'qwen, mistral'.[/dim]"
    )
    selection = Prompt.ask("Models to register", default="all")
    return _apply_selection(models, selection)


def run_configure_wizard(output: str = "configs/local.yaml") -> None:
    """Interactively gather connection info + models and write a YAML config."""
    console.print(Panel.fit(
        "[bold]Fancy LLM Router - Configuration Wizard[/bold]\n"
        "Set up an LLM connection and register models.",
        border_style="cyan",
    ))

    console.print(
        "\n[dim]Pick the provider that hosts the models (e.g. nebius, ollama,\n"
        "hyperbolic). The base URL, protocol, and API-key variable are\n"
        "pre-filled from a built-in list - just press Enter to accept. The same\n"
        "logical model can be registered from several providers; each becomes a\n"
        "deployment keyed as [bold]model@source[/bold].[/dim]\n"
    )

    provider_name = Prompt.ask(
        "Provider / source",
        choices=sorted(KNOWN_PROVIDERS.keys()),
        default="nebius",
    )
    spec = KNOWN_PROVIDERS[provider_name]

    source = Prompt.ask(
        "Source label (deployment keys become model@<source>)",
        default=provider_name,
    )

    provider = Prompt.ask(
        "API protocol the source speaks",
        choices=PROTOCOLS,
        default=spec["protocol"],
    )

    is_mock = provider == "mock"
    api_base_url = "" if is_mock else Prompt.ask("API base URL", default=spec["base_url"])

    env_var = "" if is_mock else Prompt.ask(
        "Environment variable holding the API key (blank for none)",
        default=spec["env"],
    )
    api_key_ref = "${" + env_var + "}" if env_var else None

    default_max_tokens = int(Prompt.ask("Default max output tokens", default="4096"))

    # --- Collect models: auto-discover from the provider, or paste a table ----
    parsed: List[Dict[str, Any]] = []
    can_discover = bool(api_base_url) and spec.get("openai_compatible", False)
    if can_discover and Confirm.ask(
        "Fetch the model list directly from the provider?", default=True
    ):
        api_key = os.environ.get(env_var) if env_var else None
        if env_var and not api_key:
            console.print(
                f"[yellow]{env_var} is not set in this shell; discovery may fail "
                f"for authenticated providers.[/yellow]"
            )
        try:
            with console.status("Querying the provider's /models endpoint..."):
                discovered = _discover_models(api_base_url, api_key)
            if discovered:
                parsed = _select_discovered_models(discovered)
                console.print(f"[green]Selected {len(parsed)} model(s).[/green]")
            else:
                console.print("[yellow]Provider returned no models.[/yellow]")
        except Exception as exc:  # noqa: BLE001 - surface any HTTP/parse error
            console.print(f"[yellow]Discovery failed:[/yellow] {exc}")

    if not parsed:
        # No discovery (or it failed / nothing selected): fall back to pasting.
        parsed = _collect_model_table()
    elif Confirm.ask(
        "Add pricing / throughput by pasting a table? (optional)", default=False
    ):
        # Enrich discovered models with pasted economics, matched by model id.
        extra_by_id = {p["model_id"]: p for p in _collect_model_table()}
        for m in parsed:
            extra = extra_by_id.get(m["model_id"])
            if extra:
                m.update(extra)

    if not parsed:
        console.print("[red]No models selected or parsed. Aborting.[/red]")
        sys.exit(1)

    # Preview
    table = Table(title=f"Deployments to register (source: {source})")
    table.add_column("Deployment", style="green")
    table.add_column("Model", style="cyan")
    table.add_column("$/1M in", justify="right")
    table.add_column("$/1M out", justify="right")
    table.add_column("Tok/s", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("Quant", style="magenta")
    for m in parsed:
        deployment_id = f"{_slug(m['model_id'].split('/')[-1])}@{_slug(source)}"
        tps = m.get("tokens_per_second")
        table.add_row(
            deployment_id,
            m["model_id"],
            f"${m.get('input_price_per_1m', 0.0):.2f}",
            f"${m.get('output_price_per_1m', 0.0):.2f}",
            f"{tps:g}" if tps else "-",
            f"{m.get('context_window', 4096):,}",
            m.get("quantization") or "-",
        )
    console.print(table)

    if not Confirm.ask(f"Write these {len(parsed)} models to [bold]{output}[/bold]?", default=True):
        console.print("[yellow]Aborted. No file written.[/yellow]")
        return

    # Build the models mapping keyed by a unique deployment id (model@source)
    # so the same logical model can also be registered from other sources.
    models_cfg: Dict[str, Any] = {}
    for m in parsed:
        logical_model = m["model_id"]
        name = logical_model.split("/")[-1]
        deployment_id = f"{_slug(name)}@{_slug(source)}"
        context_window = int(m.get("context_window", 4096))
        entry: Dict[str, Any] = {
            "model": logical_model,
            "provider": provider,
            "model_id": logical_model,
            "name": name,
            "source": source,
            "default_temperature": 0.7,
            "default_max_tokens": min(default_max_tokens, context_window),
            "input_token_price": m.get("input_token_price", 0.0),
            "output_token_price": m.get("output_token_price", 0.0),
            "context_window": context_window,
            "enabled": True,
            "metadata": {
                "input_price_per_1m": m.get("input_price_per_1m", 0.0),
                "output_price_per_1m": m.get("output_price_per_1m", 0.0),
            },
        }
        if api_base_url:
            entry["api_base_url"] = api_base_url
        if m.get("tokens_per_second") is not None:
            entry["tokens_per_second"] = m["tokens_per_second"]
        if m.get("quantization"):
            entry["quantization"] = m["quantization"]
        if api_key_ref:
            entry["api_key"] = api_key_ref
        models_cfg[deployment_id] = entry

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with an existing config if present so we don't clobber other sections.
    existing: Dict[str, Any] = {}
    if out_path.exists():
        try:
            existing = yaml.safe_load(out_path.read_text()) or {}
        except yaml.YAMLError:
            console.print(f"[yellow]Could not parse existing {output}; it will be replaced.[/yellow]")
            existing = {}

    config = existing if isinstance(existing, dict) else {}
    config.setdefault("app", {
        "name": "fancy_llm_router",
        "environment": "development",
        "host": "0.0.0.0",
        "port": 8000,
    })
    config.setdefault("router", {"default_strategy": "balanced"})
    config.setdefault("storage", {"backend": "sqlite", "sqlite_path": "data/metrics.db"})
    config.setdefault("models", {})
    config["models"].update(models_cfg)

    out_path.write_text(yaml.safe_dump(config, sort_keys=False, default_flow_style=False))

    console.print(f"\n[green]Wrote configuration to [bold]{out_path}[/bold][/green]")
    if api_key_ref:
        console.print(f"[dim]Remember to export your API key: export {env_var}=...[/dim]")


@click.group(invoke_without_command=True)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--config", "-c", type=str, help="Path to configuration file")
@click.option("--configure", is_flag=True, help="Run the interactive configuration wizard")
@click.option("--output", "-o", type=str, default="configs/local.yaml",
              help="Output path for the configuration wizard")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: Optional[str], configure: bool, output: str):
    """Fancy LLM Router - A production-grade LLM routing and evaluation system."""
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Stash the resolved config path so subcommands can build a router from it.
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config

    # Run the configuration wizard when requested.
    if configure:
        run_configure_wizard(output)
        ctx.exit()

    # With no subcommand and no flag, show help.
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@cli.command()
@click.option("--models", "-m", multiple=True, help="Specific models to list")
@click.pass_context
def list_models(ctx: click.Context, models: List[str]):
    """List available models."""
    router = _get_router(ctx)
    
    if models:
        # List specific models
        for model_id in models:
            model_info = router.get_model_info(model_id)
            if model_info:
                console.print(f"\n[bold]{model_info.full_id}[/bold]")
                console.print(f"  Model: {model_info.logical_model}")
                console.print(f"  Source: {model_info.source or '-'}")
                console.print(f"  Name: {model_info.name}")
                console.print(f"  Protocol: {model_info.provider.value}")
                console.print(f"  Context Window: {model_info.capabilities.context_window:,}")
                console.print(f"  Input Price: ${model_info.pricing.input_token_price * 1_000_000:,.2f}/1M tokens")
                console.print(f"  Output Price: ${model_info.pricing.output_token_price * 1_000_000:,.2f}/1M tokens")
            else:
                console.print(f"[red]Model {model_id} not found[/red]")
    else:
        # List all models
        all_models = router.list_models()
        
        if not all_models:
            console.print("[yellow]No models registered[/yellow]")
            return
        
        table = Table(title="Available Deployments")
        table.add_column("Deployment", style="green")
        table.add_column("Model", style="cyan")
        table.add_column("Source", style="yellow")
        table.add_column("Protocol", style="magenta")
        table.add_column("Context", justify="right")
        table.add_column("$/1M in", justify="right")
        table.add_column("$/1M out", justify="right")
        table.add_column("Tok/s", justify="right")
        
        for model_id in all_models:
            model_info = router.get_model_info(model_id)
            if model_info:
                tps = model_info.capabilities.tokens_per_second
                table.add_row(
                    model_info.full_id,
                    model_info.logical_model,
                    model_info.source or "-",
                    model_info.provider.value,
                    f"{model_info.capabilities.context_window:,}",
                    f"${model_info.pricing.input_token_price * 1_000_000:,.2f}",
                    f"${model_info.pricing.output_token_price * 1_000_000:,.2f}",
                    f"{tps:g}" if tps else "-",
                )
        
        console.print(table)


@cli.command()
@click.argument("prompt", required=False)
@click.option("--prompt", "-p", "prompt_option", type=str, help="Prompt text (alternative to positional argument)")
@click.option("--model", "-m", type=str, help="Specific model to use")
@click.option("--strategy", "-s", type=click.Choice([s.value for s in RoutingStrategy]), 
              default=RoutingStrategy.BALANCED.value, help="Routing strategy")
@click.option("--max-tokens", type=int, default=256, help="Maximum tokens in response")
@click.option("--temperature", type=float, default=0.7, help="Sampling temperature")
@click.option("--json", "-j", "as_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def complete(
    ctx: click.Context,
    prompt: Optional[str],
    prompt_option: Optional[str],
    model: Optional[str],
    strategy: str,
    max_tokens: int,
    temperature: float,
    as_json: bool
):
    """Generate a text completion."""
    prompt = prompt or prompt_option
    if not prompt:
        raise click.UsageError(
            "Missing prompt. Pass it as an argument or with --prompt / -p."
        )

    async def run():
        router = _get_router(ctx)
        metrics = MetricsCollector()
        
        # Create request
        request = CompletionRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        if model:
            request.model = model
        
        # Route and execute
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Routing and generating...", total=None)
            
            try:
                response = await router.execute(request, strategy=RoutingStrategy(strategy))
                decision = router._last_routing_decision
                
                progress.remove_task(task)
                
                if as_json:
                    output = {
                        "response": response.model_dump(mode="json"),
                        "model": response.model,
                    }
                    if decision:
                        output["routing_decision"] = decision.model_dump(mode="json")
                    click.echo(json.dumps(output, indent=2))
                else:
                    console.print(Panel.fit(
                        f"[bold]Model:[/bold] {response.model}\n"
                        f"[bold]Response:[/bold]\n{response.choices[0].text if response.choices else 'No response'}",
                        title="Completion Result",
                        border_style="blue"
                    ))
                    
                    if decision:
                        console.print(f"\n[dim]Routed to: {decision.full_model_id}[/dim]")
                
            except Exception as e:
                progress.remove_task(task)
                console.print(f"[red]Error:[/red] {e}")
                sys.exit(1)
    
    asyncio.run(run())


@cli.command()
@click.argument("messages", nargs=-1)
@click.option("--model", "-m", type=str, help="Specific model to use")
@click.option("--strategy", "-s", type=click.Choice([s.value for s in RoutingStrategy]), 
              default=RoutingStrategy.BALANCED.value, help="Routing strategy")
@click.option("--max-tokens", type=int, default=256, help="Maximum tokens in response")
@click.option("--temperature", type=float, default=0.7, help="Sampling temperature")
@click.option("--json", "-j", "as_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def chat(
    ctx: click.Context,
    messages: List[str],
    model: Optional[str],
    strategy: str,
    max_tokens: int,
    temperature: float,
    as_json: bool
):
    """Generate a chat completion."""
    if not messages:
        console.print("[red]Error:[/red] At least one message is required")
        sys.exit(1)
    
    async def run():
        router = _get_router(ctx)
        
        # Create chat messages
        from fancy_llm_router.schemas.requests import ChatMessage, MessageRole
        chat_messages = [
            ChatMessage(role=MessageRole.USER, content=msg)
            for msg in messages
        ]
        
        # Create request
        request = ChatRequest(
            messages=chat_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        if model:
            request.model = model
        
        # Route and execute
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Routing and generating...", total=None)
            
            try:
                response = await router.execute(request, strategy=RoutingStrategy(strategy))
                
                progress.remove_task(task)
                
                if as_json:
                    output = {
                        "response": response.model_dump(mode="json"),
                        "model": response.model,
                    }
                    click.echo(json.dumps(output, indent=2))
                else:
                    if response.choices:
                        assistant_message = response.choices[0].message.content
                        console.print(Panel.fit(
                            f"[bold]Model:[/bold] {response.model}\n"
                            f"[bold]Assistant:[/bold]\n{assistant_message}",
                            title="Chat Result",
                            border_style="green"
                        ))
                    else:
                        console.print("[yellow]No response received[/yellow]")
                
            except Exception as e:
                progress.remove_task(task)
                console.print(f"[red]Error:[/red] {e}")
                sys.exit(1)
    
    asyncio.run(run())


@cli.command()
@click.option("--strategy", "-s", type=click.Choice([s.value for s in RoutingStrategy]), 
              default=RoutingStrategy.BALANCED.value, help="Routing strategy")
@click.option("--prompt", "-p", type=str, help="Test prompt for routing")
@click.option("--models", "-m", multiple=True, help="Allowed models for routing")
@click.option("--json", "-j", "as_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def route(
    ctx: click.Context,
    strategy: str,
    prompt: Optional[str],
    models: List[str],
    as_json: bool
):
    """Test routing without executing."""
    async def run():
        router = _get_router(ctx)
        
        # Create a test request
        if prompt:
            request = CompletionRequest(prompt=prompt)
        else:
            request = CompletionRequest(prompt="Test prompt for routing")
        
        # Create criteria
        criteria = RoutingCriteria()
        if models:
            criteria.allowed_models = list(models)
        
        # Route the request
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Routing...", total=None)
            
            try:
                decision = await router.route(
                    request,
                    strategy=RoutingStrategy(strategy),
                    criteria=criteria
                )
                
                progress.remove_task(task)
                
                if as_json:
                    click.echo(json.dumps(decision.model_dump(mode="json"), indent=2))
                else:
                    table = Table(title="Routing Decision")
                    table.add_column("Property", style="cyan")
                    table.add_column("Value", style="magenta")
                    
                    table.add_row("Selected Deployment", decision.full_model_id)
                    table.add_row("Logical Model", decision.selected_model)
                    table.add_row("Protocol", decision.selected_provider)
                    table.add_row("Strategy", decision.strategy.value)
                    table.add_row("Confidence", f"{decision.confidence:.2%}")
                    table.add_row("Decision Time", f"{decision.decision_time_ms:.2f}ms")
                    table.add_row("Candidates", str(len(decision.candidates)))
                    
                    console.print(table)
                    
                    if decision.candidates:
                        console.print("\n[bold]Candidates:[/bold]")
                        for candidate in decision.candidates:
                            score = decision.candidate_scores.get(candidate, 0)
                            console.print(f"  - {candidate}: {score:.3f}")
                
            except Exception as e:
                progress.remove_task(task)
                console.print(f"[red]Error:[/red] {e}")
                sys.exit(1)
    
    asyncio.run(run())


@cli.command("benchmark")
@click.option("--root-id", "-r", required=True, help="Stable root prompt id")
@click.option("--prompt", "-p", required=True, help="Generic prompt text")
@click.option("--expected", "-e", default=None, help="Expected answer for judge")
@click.option("--deployment", "-d", default=None, help="Pin one deployment (default: all)")
@click.option("--optimize/--no-optimize", default=True, help="Refactor prompt on judge failure")
@click.option("--max-revisions", default=3, show_default=True, type=int)
@click.pass_context
def benchmark(
    ctx: click.Context,
    root_id: str,
    prompt: str,
    expected: Optional[str],
    deployment: Optional[str],
    optimize: bool,
    max_revisions: int,
):
    """Measure a prompt across deployments; judge and specialize on failure."""
    import tempfile
    import os
    from fancy_llm_router.core.benchmark_service import BenchmarkService
    from fancy_llm_router.core.config_loader import get_storage_db_path, load_config
    from fancy_llm_router.core.prompt_registry import PromptRegistry

    router = _get_router(ctx)
    config_path = ctx.obj.get("config_path") if ctx.obj else None
    config_dict = {}
    if config_path:
        config_dict = load_config(config_path)
    db_path = get_storage_db_path(config_dict)
    registry = PromptRegistry(db_path=db_path)
    registry.initialize()
    router.set_prompt_registry(registry)
    service = BenchmarkService(router=router, registry=registry)

    async def run():
        if deployment:
            from fancy_llm_router.schemas.prompts import BenchmarkMeasureRequest

            envelope = await service.measure_deployment(
                BenchmarkMeasureRequest(
                    root_id=root_id,
                    prompt=prompt,
                    expected_answer=expected,
                    deployment_id=deployment,
                    optimize=optimize,
                    max_revisions=max_revisions,
                )
            )
            console.print(envelope.dict(by_alias=True))
        else:
            result = await service.measure_all_deployments(
                root_id=root_id,
                prompt=prompt,
                expected_answer=expected,
                optimize=optimize,
                max_revisions=max_revisions,
            )
            console.print_json(data=result)

    asyncio.run(run())


@cli.command()
def evaluate():
    """Run evaluation benchmarks."""
    console.print("[yellow]Evaluation feature coming soon![/yellow]")


@cli.command()
@click.option("--output", "-o", type=str, default="configs/local.yaml",
              help="Path to write the configuration file")
def configure(output: str):
    """Interactively configure an LLM connection and register models."""
    run_configure_wizard(output)


@cli.command()
@click.pass_context
def serve(ctx: click.Context):
    """Start the API server."""
    from fancy_llm_router.api.server import create_app
    import uvicorn
    
    config_path = ctx.obj.get("config_path") if ctx.obj else None
    app = create_app(config_path=config_path)
    console.print("[bold green]Starting LLM Router API server...[/bold green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
