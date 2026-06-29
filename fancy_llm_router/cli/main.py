"""Main CLI for the LLM Router."""

import argparse
import asyncio
import json
import logging
import sys
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from fancy_llm_router.core.router import LLMRouter, RoutingStrategy
from fancy_llm_router.metrics.collector import MetricsCollector
from fancy_llm_router.schemas.requests import CompletionRequest, ChatRequest
from fancy_llm_router.schemas.routing import RoutingCriteria

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--config", "-c", type=str, help="Path to configuration file")
def cli(verbose: bool, config: Optional[str]):
    """Fancy LLM Router - A production-grade LLM routing and evaluation system."""
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Load configuration if provided
    if config:
        # In a real implementation, this would load the config file
        pass


@cli.command()
@click.option("--models", "-m", multiple=True, help="Specific models to list")
def list_models(models: List[str]):
    """List available models."""
    router = LLMRouter()
    
    if models:
        # List specific models
        for model_id in models:
            model_info = router.get_model_info(model_id)
            if model_info:
                console.print(f"\n[bold]{model_info.full_id}[/bold]")
                console.print(f"  Name: {model_info.name}")
                console.print(f"  Provider: {model_info.provider.value}")
                console.print(f"  Context Window: {model_info.capabilities.context_window}")
                console.print(f"  Input Price: ${model_info.pricing.input_token_price:.6f}/token")
                console.print(f"  Output Price: ${model_info.pricing.output_token_price:.6f}/token")
            else:
                console.print(f"[red]Model {model_id} not found[/red]")
    else:
        # List all models
        all_models = router.list_models()
        
        if not all_models:
            console.print("[yellow]No models registered[/yellow]")
            return
        
        table = Table(title="Available Models")
        table.add_column("ID", style="cyan")
        table.add_column("Provider", style="magenta")
        table.add_column("Context Window", justify="right")
        table.add_column("Input Price", justify="right")
        table.add_column("Output Price", justify="right")
        
        for model_id in all_models:
            model_info = router.get_model_info(model_id)
            if model_info:
                table.add_row(
                    model_id,
                    model_info.provider.value,
                    str(model_info.capabilities.context_window),
                    f"${model_info.pricing.input_token_price:.6f}",
                    f"${model_info.pricing.output_token_price:.6f}",
                )
        
        console.print(table)


@cli.command()
@click.argument("prompt")
@click.option("--model", "-m", type=str, help="Specific model to use")
@click.option("--strategy", "-s", type=click.Choice([s.value for s in RoutingStrategy]), 
              default=RoutingStrategy.BALANCED.value, help="Routing strategy")
@click.option("--max-tokens", type=int, default=256, help="Maximum tokens in response")
@click.option("--temperature", type=float, default=0.7, help="Sampling temperature")
@click.option("--json", "-j", is_flag=True, help="Output in JSON format")
def complete(
    prompt: str,
    model: Optional[str],
    strategy: str,
    max_tokens: int,
    temperature: float,
    json: bool
):
    """Generate a text completion."""
    async def run():
        router = LLMRouter()
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
                if model:
                    # Use specific model
                    decision = None
                    response = await router.execute(request, strategy=RoutingStrategy(strategy))
                else:
                    # Route to best model
                    decision = await router.route(request, strategy=RoutingStrategy(strategy))
                    response = await router.execute(request, strategy=RoutingStrategy(strategy))
                
                progress.remove_task(task)
                
                if json:
                    output = {
                        "response": response.dict(),
                        "model": response.model,
                    }
                    if decision:
                        output["routing_decision"] = decision.dict()
                    console.print(json.dumps(output, indent=2))
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
@click.option("--json", "-j", is_flag=True, help="Output in JSON format")
def chat(
    messages: List[str],
    model: Optional[str],
    strategy: str,
    max_tokens: int,
    temperature: float,
    json: bool
):
    """Generate a chat completion."""
    if not messages:
        console.print("[red]Error:[/red] At least one message is required")
        sys.exit(1)
    
    async def run():
        router = LLMRouter()
        
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
                if model:
                    # Use specific model
                    response = await router.execute(request, strategy=RoutingStrategy(strategy))
                else:
                    # Route to best model
                    decision = await router.route(request, strategy=RoutingStrategy(strategy))
                    response = await router.execute(request, strategy=RoutingStrategy(strategy))
                
                progress.remove_task(task)
                
                if json:
                    output = {
                        "response": response.dict(),
                        "model": response.model,
                    }
                    console.print(json.dumps(output, indent=2))
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
@click.option("--json", "-j", is_flag=True, help="Output in JSON format")
def route(
    strategy: str,
    prompt: Optional[str],
    models: List[str],
    json: bool
):
    """Test routing without executing."""
    async def run():
        router = LLMRouter()
        
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
                
                if json:
                    console.print(json.dumps(decision.dict(), indent=2))
                else:
                    table = Table(title="Routing Decision")
                    table.add_column("Property", style="cyan")
                    table.add_column("Value", style="magenta")
                    
                    table.add_row("Selected Model", decision.full_model_id)
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


@cli.command()
def evaluate():
    """Run evaluation benchmarks."""
    console.print("[yellow]Evaluation feature coming soon![/yellow]")


@cli.command()
def serve():
    """Start the API server."""
    from fancy_llm_router.api.server import create_app
    import uvicorn
    
    app = create_app()
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
