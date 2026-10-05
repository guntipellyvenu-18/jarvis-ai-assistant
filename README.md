# jarvis-ai-assistant

A production-ready Python assistant built on top of Ollama with multiple callable tools for system inspection, file access, and process/network diagnostics.

## Features

- System diagnostics
- Process listing
- File reading
- Directory listing
- Network summary
- Structured logging and safe file handling
- Configurable model selection via environment variables
- Tool-call loop with bounded retries

## Quick start

1. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start Ollama locally and ensure a model is available:

```bash
ollama pull llama3.1
```

3. Run the assistant:

```bash
python jarvis_app.py "Check my system health and list the top 5 running processes"
```

## Environment variables

```bash
export OLLAMA_MODEL=llama3.1
export JARVIS_LOG_LEVEL=INFO
```

## Example tool usage

- "Show my CPU and memory usage"
- "List the most CPU-intensive processes"
- "Read the last 50 lines of app logs"
- "List the files in the current directory"

## Files

- `jarvis_app.py` — main JARVIS implementation
- `requirements.txt` — Python dependencies
- `tests/test_jarvis_tools.py` — lightweight verification tests

## Notes

- File reads are intentionally limited to safe sizes and line counts.
- Tool calls are bounded to prevent runaway loops.
- All tool results are returned as structured JSON to the model.
