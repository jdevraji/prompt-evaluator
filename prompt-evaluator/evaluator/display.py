"""Rich-powered terminal display for evaluation results."""

import json
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich import box

from evaluator.models import FinalVerdict


console = Console()

BANNER = r"""
[bold cyan]╔══════════════════════════════════════════════════════════════╗
║             🧠  AI Prompt Evaluator  v1.0                    ║
║                  Mini RLHF Tool                              ║
╚══════════════════════════════════════════════════════════════╝[/bold cyan]
"""


def show_banner():
    """Display the app banner."""
    console.print(BANNER)


def show_input_summary(prompt: str, response_a: str, response_b: str):
    """Show a summary of the inputs."""
    console.print()
    console.print(Panel(
        f"[dim]{prompt[:200]}{'...' if len(prompt) > 200 else ''}[/dim]",
        title="[bold yellow]📝 Original Prompt[/bold yellow]",
        border_style="yellow",
        padding=(0, 2),
    ))

    cols = Columns([
        Panel(
            f"[dim]{response_a[:150]}{'...' if len(response_a) > 150 else ''}[/dim]",
            title="[bold blue]Response A[/bold blue]",
            border_style="blue",
            width=40,
            padding=(0, 1),
        ),
        Panel(
            f"[dim]{response_b[:150]}{'...' if len(response_b) > 150 else ''}[/dim]",
            title="[bold magenta]Response B[/bold magenta]",
            border_style="magenta",
            width=40,
            padding=(0, 1),
        ),
    ], equal=True, expand=True)
    console.print(cols)


def show_heuristic_scores(verdict: FinalVerdict):
    """Display the heuristic scoring breakdown table."""
    console.print()
    table = Table(
        title="[bold]📊 Heuristic Scoring Breakdown[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white on dark_blue",
        border_style="blue",
        padding=(0, 1),
    )

    table.add_column("Dimension", style="bold", min_width=18)
    table.add_column("Response A", justify="center", min_width=12)
    table.add_column("Response B", justify="center", min_width=12)
    table.add_column("Winner", justify="center", min_width=8)

    for dim in verdict.heuristic_result.dimensions:
        # Color the scores
        a_style = "green" if dim.score_a > dim.score_b else ("red" if dim.score_a < dim.score_b else "yellow")
        b_style = "green" if dim.score_b > dim.score_a else ("red" if dim.score_b < dim.score_a else "yellow")

        if dim.score_a > dim.score_b:
            winner_text = "[green bold]A ✓[/green bold]"
        elif dim.score_b > dim.score_a:
            winner_text = "[magenta bold]B ✓[/magenta bold]"
        else:
            winner_text = "[yellow]Tie[/yellow]"

        # Score bars
        a_bar = _score_bar(dim.score_a)
        b_bar = _score_bar(dim.score_b)

        table.add_row(
            dim.dimension,
            f"[{a_style}]{dim.score_a:.1f}[/{a_style}] {a_bar}",
            f"[{b_style}]{dim.score_b:.1f}[/{b_style}] {b_bar}",
            winner_text,
        )

    # Totals row
    table.add_section()
    h = verdict.heuristic_result
    a_total_style = "green bold" if h.total_a >= h.total_b else "red bold"
    b_total_style = "green bold" if h.total_b >= h.total_a else "red bold"
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[{a_total_style}]{h.total_a:.1f}/60[/{a_total_style}]",
        f"[{b_total_style}]{h.total_b:.1f}/60[/{b_total_style}]",
        f"[bold]{h.winner}[/bold]",
    )

    console.print(table)


def _score_bar(score: float) -> str:
    """Generate a mini bar chart for a score (0-10)."""
    filled = int(score)
    empty = 10 - filled
    return f"[green]{'█' * filled}[/green][dim]{'░' * empty}[/dim]"


def show_llm_result(verdict: FinalVerdict):
    """Display the LLM evaluation results."""
    llm = verdict.llm_result
    if llm is None:
        console.print()
        console.print(Panel(
            "[yellow]LLM evaluation was skipped or unavailable.\n"
            "Set OPENROUTER_API_KEY to enable AI-powered evaluation.[/yellow]",
            title="[bold yellow]🤖 LLM Evaluation[/bold yellow]",
            border_style="yellow",
        ))
        return

    console.print()

    # Strengths and weaknesses
    content_lines = []

    content_lines.append(f"[bold]LLM Scores:[/bold]  Response A: [cyan]{llm.score_a:.1f}/10[/cyan]  |  Response B: [cyan]{llm.score_b:.1f}/10[/cyan]")
    content_lines.append("")

    if llm.strengths_a:
        content_lines.append("[bold green]✅ Response A Strengths:[/bold green]")
        for s in llm.strengths_a:
            content_lines.append(f"   • {s}")
        content_lines.append("")

    if llm.weaknesses_a:
        content_lines.append("[bold red]❌ Response A Weaknesses:[/bold red]")
        for w in llm.weaknesses_a:
            content_lines.append(f"   • {w}")
        content_lines.append("")

    if llm.strengths_b:
        content_lines.append("[bold green]✅ Response B Strengths:[/bold green]")
        for s in llm.strengths_b:
            content_lines.append(f"   • {s}")
        content_lines.append("")

    if llm.weaknesses_b:
        content_lines.append("[bold red]❌ Response B Weaknesses:[/bold red]")
        for w in llm.weaknesses_b:
            content_lines.append(f"   • {w}")
        content_lines.append("")

    if llm.reasoning:
        content_lines.append(f"[bold]💬 Reasoning:[/bold] {llm.reasoning}")

    console.print(Panel(
        "\n".join(content_lines),
        title="[bold cyan]🤖 LLM Evaluation (DeepSeek R1)[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))


def show_verdict(verdict: FinalVerdict):
    """Display the final verdict banner."""
    console.print()

    if verdict.winner == "TIE":
        color = "yellow"
        emoji = "🤝"
        label = "IT'S A TIE!"
        detail = "Both responses are closely matched."
    elif verdict.winner == "A":
        color = "green"
        emoji = "🏆"
        label = "RESPONSE A WINS!"
        detail = "Response A is the stronger answer."
    else:
        color = "magenta"
        emoji = "🏆"
        label = "RESPONSE B WINS!"
        detail = "Response B is the stronger answer."

    verdict_text = (
        f"[bold {color}]{emoji}  {label}[/bold {color}]\n\n"
        f"[bold]Final Scores:[/bold]  "
        f"A = [cyan]{verdict.score_a:.2f}[/cyan]  |  "
        f"B = [cyan]{verdict.score_b:.2f}[/cyan]\n"
        f"[bold]Confidence:[/bold]  {verdict.confidence:.0f}%\n\n"
        f"[dim]{detail}[/dim]"
    )

    console.print(Panel(
        verdict_text,
        title=f"[bold {color}]══ FINAL VERDICT ══[/bold {color}]",
        border_style=color,
        padding=(1, 4),
    ))

    # Detailed reasoning
    if verdict.reasoning:
        console.print()
        console.print(Panel(
            f"[dim]{verdict.reasoning}[/dim]",
            title="[bold]📋 Detailed Reasoning[/bold]",
            border_style="dim",
            padding=(0, 2),
        ))


def show_json_output(verdict: FinalVerdict):
    """Output results as JSON for piping."""
    output = {
        "winner": verdict.winner,
        "confidence": verdict.confidence,
        "score_a": verdict.score_a,
        "score_b": verdict.score_b,
        "heuristic": {
            "total_a": verdict.heuristic_result.total_a,
            "total_b": verdict.heuristic_result.total_b,
            "winner": verdict.heuristic_result.winner,
            "dimensions": [
                {
                    "name": d.dimension,
                    "score_a": d.score_a,
                    "score_b": d.score_b,
                }
                for d in verdict.heuristic_result.dimensions
            ],
        },
        "reasoning": verdict.reasoning,
    }

    if verdict.llm_result:
        output["llm"] = {
            "score_a": verdict.llm_result.score_a,
            "score_b": verdict.llm_result.score_b,
            "winner": verdict.llm_result.winner,
            "strengths_a": verdict.llm_result.strengths_a,
            "strengths_b": verdict.llm_result.strengths_b,
            "weaknesses_a": verdict.llm_result.weaknesses_a,
            "weaknesses_b": verdict.llm_result.weaknesses_b,
            "reasoning": verdict.llm_result.reasoning,
        }

    console.print_json(json.dumps(output, indent=2))
