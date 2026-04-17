"""CLI entry point for the AI Prompt Evaluator."""

import json
import sys

import click
from rich.console import Console
from rich.prompt import Prompt
from rich.status import Status
from dotenv import load_dotenv

from evaluator import __version__
from evaluator.models import EvaluationInput
from evaluator import heuristic_scorer
from evaluator import llm_scorer
from evaluator import combiner
from evaluator import display


console = Console()


def _read_multiline(prompt_text: str) -> str:
    """Read multi-line input from the user. End with an empty line."""
    console.print(f"\n[bold cyan]{prompt_text}[/bold cyan]")
    console.print("[dim](Paste your text, then press Enter twice to finish)[/dim]")
    lines = []
    empty_count = 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            empty_count += 1
            if empty_count >= 2:
                break
            lines.append(line)
        else:
            empty_count = 0
            lines.append(line)
    return "\n".join(lines).strip()


def _load_from_file(filepath: str) -> EvaluationInput:
    """Load evaluation input from a JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[red bold]Error:[/red bold] File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red bold]Error:[/red bold] Invalid JSON: {e}")
        sys.exit(1)

    required_keys = {"prompt", "response_a", "response_b"}
    missing = required_keys - set(data.keys())
    if missing:
        console.print(f"[red bold]Error:[/red bold] Missing keys in JSON: {', '.join(missing)}")
        console.print("[dim]Required keys: prompt, response_a, response_b[/dim]")
        sys.exit(1)

    return EvaluationInput(
        prompt=data["prompt"],
        response_a=data["response_a"],
        response_b=data["response_b"],
    )


@click.command()
@click.option("--file", "-f", "filepath", type=str, default=None,
              help="Path to a JSON file containing prompt, response_a, and response_b.")
@click.option("--model", "-m", type=str, default=None,
              help="Model to use for LLM evaluation (default: deepseek/deepseek-r1:free).")
@click.option("--key", "-k", type=str, default=None,
              help="OpenRouter API key (overrides OPENROUTER_API_KEY env var).")
@click.option("--no-llm", is_flag=True, default=False,
              help="Skip LLM evaluation, use heuristic scoring only.")
@click.option("--json-output", "--json", "json_mode", is_flag=True, default=False,
              help="Output results as JSON for piping.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show detailed scoring breakdown.")
@click.version_option(version=__version__, prog_name="prompt-evaluator")
def main(filepath, model, key, no_llm, json_mode, verbose):
    """🧠 AI Prompt Evaluator — Compare two AI responses and find the better one.

    Run interactively (default) or provide a JSON file with --file.

    \b
    Example JSON file format:
    {
      "prompt": "Your question here",
      "response_a": "First AI response",
      "response_b": "Second AI response"
    }
    \b
    Example with model and key:
      prompt-evaluator --model deepseek/deepseek-r1:free --key sk-or-v1-xxx --file input.json
    """
    # Load env vars (for API key)
    load_dotenv()

    if not json_mode:
        display.show_banner()

    # Get input
    if filepath:
        evaluation_input = _load_from_file(filepath)
        if not json_mode:
            console.print(f"[green]✓[/green] Loaded input from [bold]{filepath}[/bold]")
    else:
        # Interactive mode
        if json_mode:
            console.print("[red]Error: Interactive mode is not compatible with --json. Use --file instead.[/red]")
            sys.exit(1)

        prompt = _read_multiline("📝 Enter the original prompt:")
        if not prompt:
            console.print("[red]Error: Prompt cannot be empty.[/red]")
            sys.exit(1)

        response_a = _read_multiline("🅰️  Enter Response A:")
        if not response_a:
            console.print("[red]Error: Response A cannot be empty.[/red]")
            sys.exit(1)

        response_b = _read_multiline("🅱️  Enter Response B:")
        if not response_b:
            console.print("[red]Error: Response B cannot be empty.[/red]")
            sys.exit(1)

        evaluation_input = EvaluationInput(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
        )

    # Show input summary
    if not json_mode:
        display.show_input_summary(
            evaluation_input.prompt,
            evaluation_input.response_a,
            evaluation_input.response_b,
        )

    # Step 1: Heuristic scoring
    if not json_mode:
        console.print()
        with Status("[bold blue]Running heuristic analysis...[/bold blue]", spinner="dots"):
            heuristic_result = heuristic_scorer.score(evaluation_input)
    else:
        heuristic_result = heuristic_scorer.score(evaluation_input)

    # Step 2: LLM scoring (if enabled)
    llm_result = None
    if not no_llm:
        # Apply --key override if provided
        if key:
            import os
            os.environ["OPENROUTER_API_KEY"] = key

        model_display = model or llm_scorer.MODEL
        if not json_mode:
            with Status(f"[bold cyan]Querying {model_display} via OpenRouter...[/bold cyan]", spinner="dots"):
                llm_result = llm_scorer.score(evaluation_input, model_override=model)
            if llm_result is None:
                console.print("[yellow]⚠  LLM evaluation unavailable — falling back to heuristic only.[/yellow]")
        else:
            llm_result = llm_scorer.score(evaluation_input, model_override=model)

    # Step 3: Combine scores
    verdict = combiner.combine(heuristic_result, llm_result)

    # Step 4: Display results
    if json_mode:
        display.show_json_output(verdict)
    else:
        display.show_heuristic_scores(verdict)
        display.show_llm_result(verdict)
        display.show_verdict(verdict)
        console.print()


if __name__ == "__main__":
    main()
