# 🧠 AI Prompt Evaluator (Mini RLHF Tool)

A CLI tool that compares two AI responses to a prompt and determines which one is better — using both local heuristic scoring and LLM-powered evaluation via OpenRouter (DeepSeek R1).

## Quick Start

```bash
# Install
pip install -e .

# Run interactively
prompt-evaluator

# Run with a file
prompt-evaluator --file examples/sample_input.json

# Specify model and API key directly
prompt-evaluator --model deepseek/deepseek-r1:free --key sk-or-v1-xxx --file examples/sample_input.json
```

## Setup

### 1. Install dependencies

```bash
cd prompt-evaluator
pip install -e .
```

### 2. Set up your API key (optional but recommended)

Get a free API key from [openrouter.ai](https://openrouter.ai) and set it:

```bash
# Option A: Environment variable
export OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Option B: .env file
cp .env.example .env
# Edit .env and add your key
```

> Without an API key, the tool falls back to heuristic-only scoring.

## Usage

### Interactive Mode (default)

```bash
prompt-evaluator
```

You'll be prompted to paste:
1. The original prompt
2. Response A
3. Response B

Press Enter twice after each to submit.

### File Mode

```bash
prompt-evaluator --file input.json
```

JSON file format:
```json
{
  "prompt": "Your question here",
  "response_a": "First AI response",
  "response_b": "Second AI response"
}
```

### Options

| Flag | Description |
|------|-------------|
| `--file`, `-f` | Load input from a JSON file |
| `--model`, `-m` | Model to use for LLM evaluation (default: `deepseek/deepseek-r1:free`) |
| `--key`, `-k` | OpenRouter API key (overrides `OPENROUTER_API_KEY` env var) |
| `--no-llm` | Skip LLM evaluation (heuristic only) |
| `--json` | Output results as JSON |
| `--verbose`, `-v` | Show detailed breakdown |
| `--version` | Show version |
| `--help` | Show help |

### Examples

```bash
# Use a different model
prompt-evaluator --model openai/gpt-4o-mini --file input.json

# Pass API key directly (no env var needed)
prompt-evaluator --key sk-or-v1-your-key --file input.json

# Combine both
prompt-evaluator -m deepseek/deepseek-r1:free -k sk-or-v1-xxx -f input.json

# Heuristic only (no API key needed)
prompt-evaluator --no-llm --file input.json
```

## How Scoring Works

### Heuristic Scoring (Local)
Evaluates on 6 dimensions (0-10 each, 60 total):

| Dimension | What it measures |
|-----------|-----------------|
| Length & Detail | Word count relative to prompt complexity |
| Structure | Lists, paragraphs, headers, formatting elements |
| Keyword Relevance | How many prompt keywords appear in the response |
| Formatting | Code blocks, punctuation, capitalization |
| Specificity | Concrete language vs. vague filler words |
| Coherence | Transition words, sentence flow, logical structure |

### LLM Scoring (via OpenRouter)
Uses DeepSeek R1 Free to evaluate helpfulness, accuracy, clarity, completeness, and relevance.

### Final Score
- **With LLM**: 40% heuristic + 60% LLM
- **Without LLM**: 100% heuristic

## License

MIT
