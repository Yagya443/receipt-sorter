# Receipt Sorter

Receipt Sorter uses a local Ollama model to extract receipt details and organize
your spending. It includes a simple command-line sorter for `.txt` files and a
Streamlit app with PDF uploads, a receipt ledger, vendor rules,
spending dashboards, and CSV/Excel exports. Receipt data is stored locally in
`receipts.db`.

## Requirements

- Python 3.9 or newer
- [Ollama](https://ollama.com/) installed and running locally
- The `qwen3:4b` model pulled into Ollama (or another model selected in the app or `sorter.py`)

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

## Run the Streamlit app

The app supports `.txt` files, text-based PDFs, and pasted receipt text:

```bash
python -m streamlit run app_pro.py
```

Use the Upload tab to add receipts, the Dashboard to review spending, and Review
& Edit to correct extracted data. The app flags uncertain totals or dates for
review. Scanned PDFs and receipt images are not supported yet.

## Run the command-line sorter

Put one plain-text receipt per `.txt` file in `receipts/`, then run:

```bash
python sorter.py
```

The script uses Ollama's JSON response format, retries once when the response is
invalid, then writes `output.csv` and prints category totals. Its categories are
`food`, `travel`, `software`, `office`, and `other`.

To use a different installed Ollama model, change the model selector in the app
or the `MODEL` value at the top of `sorter.py`.

## Run tests

```bash
python -m unittest
```
