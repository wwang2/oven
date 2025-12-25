import click
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Dict, Any, List

from oven_core.backends.local import LocalBackend
from oven_core.runtime import RunResult

console = Console()


def get_backend(backend_name: str):
    if backend_name == "local":
        return LocalBackend()
    elif backend_name == "modal":
        from oven_core.backends.modal_backend import ModalBackend
        return ModalBackend()
    else:
        raise click.BadParameter(f"Unknown backend: {backend_name}")


def parse_inputs(input_items: tuple) -> Dict[str, Any]:
    """Parse input key=value pairs into a dictionary."""
    inputs = {}
    for item in input_items:
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                inputs[k] = json.loads(v)
            except json.JSONDecodeError:
                inputs[k] = v
    return inputs

@click.group()
def cli():
    """Oven: Slurm-like serverless workload orchestrator."""
    pass

@cli.command()
def deploy():
    """Deploy the Modal app."""
    try:
        from oven_core.backends.modal_backend import app
        console.print("[yellow]Deploying Modal app...[/yellow]")
        app.deploy()
        console.print("[green]App deployed successfully![/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@cli.command()
@click.argument("task_name")
@click.option("--backend", default="local", help="Execution backend (local, modal)")
@click.option("--input", "-i", multiple=True, help="Input parameters in key=value format")
def run(task_name, backend, input):
    """Run a single workload."""
    inputs = parse_inputs(input)
    
    try:
        be = get_backend(backend)
        console.print(f"[yellow]Submitting task [bold]{task_name}[/bold] to [bold]{backend}[/bold] backend...[/yellow]")
        run_id = be.submit(task_name, inputs)
        console.print(f"[green]Run submitted! ID: [bold]{run_id}[/bold][/green]")
        
        # For local execution, it's already finished since it's synchronous
        if backend == "local":
            result = be.get_status(run_id)
            if result.status == "succeeded":
                console.print("[green]Run completed successfully.[/green]")
                console.print(f"Artifacts: {result.artifact_path}")
                console.print("Metadata:", result.metadata)
            else:
                console.print(f"[red]Run failed: {result.error}[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.argument("task_name")
@click.option("--backend", default="local", help="Execution backend (local, modal)")
@click.option("--param", "-p", required=True, help="Parameter name to sweep over")
@click.option("--values", "-v", required=True, help="JSON array of values to map over")
@click.option("--input", "-i", multiple=True, help="Additional fixed inputs in key=value format")
@click.option("--concurrency", "-c", default=None, type=int, help="Maximum concurrent tasks")
def map(task_name, backend, param, values, input, concurrency):
    """
    Map a task over multiple parameter values in parallel.
    
    Example:
        oven map demo_task -p n -v "[10, 20, 50, 100]"
        oven map train_model -p learning_rate -v "[0.001, 0.01, 0.1]" -i epochs=100
    """
    try:
        param_values = json.loads(values)
        if not isinstance(param_values, list):
            raise click.BadParameter("Values must be a JSON array")
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"Invalid JSON for values: {e}")
    
    # Parse fixed inputs
    fixed_inputs = parse_inputs(input)
    
    # Build inputs list
    inputs_list = [{**fixed_inputs, param: v} for v in param_values]
    
    try:
        be = get_backend(backend)
        console.print(f"[yellow]Mapping task [bold]{task_name}[/bold] over {len(inputs_list)} inputs on [bold]{backend}[/bold]...[/yellow]")
        
        succeeded = 0
        failed = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running {len(inputs_list)} tasks...", total=len(inputs_list))
            
            for run_id, result in be.map(task_name, inputs_list, concurrency):
                if result.status == "succeeded":
                    succeeded += 1
                    progress.console.print(f"  [green]✓[/green] {run_id}: {result.metadata.get(param, 'done')}")
                else:
                    failed += 1
                    progress.console.print(f"  [red]✗[/red] {run_id}: {result.error[:50] if result.error else 'failed'}...")
                progress.advance(task)
        
        console.print()
        console.print(f"[bold]Results:[/bold] {succeeded} succeeded, {failed} failed")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()

@cli.command()
@click.argument("run_id")
@click.option("--backend", default="local", help="Execution backend")
def status(run_id, backend):
    """Check the status of a run."""
    try:
        be = get_backend(backend)
        result = be.get_status(run_id)
        
        table = Table(title=f"Run Status: {run_id}")
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        
        table.add_row("Status", result.status)
        if result.artifact_path:
            table.add_row("Artifact Path", result.artifact_path)
        if result.error:
            table.add_row("Error", result.error)
        
        console.print(table)
        if result.metadata:
            console.print("\n[bold]Metadata:[/bold]")
            console.print_json(data=result.metadata)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@cli.command()
@click.argument("run_id")
@click.option("--backend", default="local", help="Execution backend")
@click.option("--dest", "-d", default="downloads", help="Destination directory")
def fetch(run_id, backend, dest):
    """Fetch artifacts for a run."""
    try:
        be = get_backend(backend)
        dest_path = Path(dest) / run_id
        console.print(f"[yellow]Fetching artifacts for run [bold]{run_id}[/bold]...[/yellow]")
        final_path = be.fetch_artifacts(run_id, dest_path)
        console.print(f"[green]Artifacts saved to: [bold]{final_path}[/bold][/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    cli()

