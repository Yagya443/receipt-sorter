# Local Receipt Sorter

This small Python tool reads `.txt` receipts from `receipts/`, asks a local Ollama
model to extract the vendor, date, total, and category, then writes the results to
`output.csv` and prints spending totals for each category.

## Requirements

- Python 3.9 or newer
- [Ollama](https://ollama.com/) installed and running locally
- The `qwen3:4b` model pulled into Ollama (or another model selected in `sorter.py`)

Install the Python dependency:

```bash
python -m pip install -r requirements.txt
```

Download the default model if needed:

```bash
ollama pull qwen3:4b
```

Ollama must be running locally when you execute the sorter. If its service is
not already running, start it in a separate terminal:

```bash
ollama serve
```

## Run

Put one plain-text receipt per `.txt` file in the `receipts/` folder, then run:

```bash
python sorter.py
```

The script uses Ollama's JSON response format and disables model thinking. It
removes any `<think>...</think>` block if one appears and retries once when the
response is invalid. The output CSV contains `vendor`, `date`, `total`, and
`category`; totals are written as plain numbers.

To use a different installed Ollama model, change the `MODEL` value at the top
of `sorter.py`.

## Receipt categories

Each receipt is assigned one category: `food`, `travel`, `software`, `office`,
or `other`.
