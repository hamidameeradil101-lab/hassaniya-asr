"""Copy train_whisper.py into the Kaggle notebook's section-3 cells.

The notebook writes the script in pieces: one `%%writefile` cell followed by `%%writefile -a` cells.
This splits train_whisper.py at fixed markers, replaces each piece's cell source (markdown cells are
left alone), and checks that the cells rebuild the script.

    python tools/sync_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "train_whisper.py"
NOTEBOOK = ROOT / "notebooks" / "hassaniya-asr-whisper-medium-fine-tuning.ipynb"
TARGET = "/kaggle/working/train_whisper.py"

# Each piece starts at one of these lines; the first piece starts at the top of the file
MARKERS = [
    "def parse_args():",
    "# Identical copy of normalize_arabic",
    "def find_data_dir(",
    "@dataclass",
    "def make_compute_metrics(",
    "def dry_run(",
    "def main():",
    "    training_args = Seq2SeqTrainingArguments(",
    "def score_test(",
    "def write_results(",
]


def split_script(script):
    starts = [0] + [script.index(m) for m in MARKERS]
    if starts != sorted(starts):
        raise SystemExit("Markers are out of order in train_whisper.py")
    return [script[a:b] for a, b in zip(starts, starts[1:] + [len(script)])]


def to_source(text):
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def main():
    script = SCRIPT.read_text(encoding="utf-8")
    pieces = split_script(script)

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if "".join(c["source"]).startswith("%%writefile")]
    if len(cells) != len(pieces):
        raise SystemExit(f"Notebook has {len(cells)} %%writefile cells, script splits into {len(pieces)} pieces")

    for i, (cell, piece) in enumerate(zip(cells, pieces)):
        body = piece.rstrip("\n")
        # Appended pieces start with a blank line so an append never merges two lines
        cell["source"] = to_source(f"%%writefile {TARGET}\n{body}" if i == 0 else f"%%writefile -a {TARGET}\n\n{body}")

    rebuilt = "".join("".join(c["source"]).partition("\n")[2] + "\n" for c in cells)
    code_lines = lambda text: [line for line in text.split("\n") if line.strip()]
    if code_lines(rebuilt) != code_lines(script):
        raise SystemExit("Rebuilt script differs from train_whisper.py")

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Synced {len(pieces)} cells into {NOTEBOOK.name}")


if __name__ == "__main__":
    main()
