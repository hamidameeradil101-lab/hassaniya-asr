# Hassaniya Arabic Speech Recognition — Whisper Fine-Tuning Baseline

Fine-tuning OpenAI's [Whisper-medium](https://huggingface.co/openai/whisper-medium) to transcribe **Hassaniya Arabic**, the Arabic dialect of Mauritania. Almost no public speech data exists for it.

This is a **prototype and baseline**. It was built to see how far a pretrained multilingual model gets on the ~3 hours of public Hassaniya speech available, and to set up an evaluation pipeline that can be trusted before investing in real data collection.

## Results

All numbers are on a held-out test set of 289 clips (14 minutes of audio), scored once with the checkpoint chosen on the validation set. Text is normalized before scoring (see [Text normalization](#text-normalization)).

| Model | WER | CER |
|---|---|---|
| Whisper-medium, no fine-tuning (zero-shot) | 85.5% | 46.7% |
| **Whisper-medium, fine-tuned (this repo)** | **58.8%** | **22.5%** |

Both rows use the same decoding settings (greedy, with guards against looping output; see [Decoding guards](#decoding-guards)). Out of the box, Whisper gets most Hassaniya words wrong. Fine-tuning cuts the word error rate by about a third and more than halves the character error rate.

**By source dataset:**

| Source | Test clips | WER | CER |
|---|---|---|---|
| A — `hassaniya-parler-tts` | 102 | 53.5% | 16.2% |
| B — `abc.hassaniya_ASR` | 168 | 61.0% | 25.0% |
| C — `Hassaniya-speech-dataset` | 19 | 49.5% | 17.3% |

Dataset B is the most realistic estimate. Only B has episode information, so its test clips come from recordings never heard during training. A and C are split clip by clip, so their test sets may share speakers with training. C's 19 clips are also too few for a stable number.

**Training curve** (validation set, 268 clips):

| Epoch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Val WER | 64.5% | 63.5% | 60.9% | 60.7% | **60.6%** |
| Val CER | 24.5% | 23.5% | 22.7% | 22.3% | **22.3%** |
| Val loss | 0.85 | **0.80** | 0.83 | 0.88 | 0.91 |

Epoch 5 was selected. Validation loss bottomed out at epoch 2 while WER kept improving until epoch 5: the model grows over-confident on its mistakes (higher loss) even as its top guesses get better. Selecting the checkpoint by WER rather than loss mattered here.

### Takeaways
- **Data is the bottleneck.** Training loss fell to ~0.02 while validation WER improved by only 4 points after the first epoch. More epochs or hyperparameter tuning on this corpus won't change the picture much; more and better-transcribed data will.
- **Word error rate overstates the errors.** Hassaniya has no standard spelling. Many "errors" are spelling variants (`هاذ`/`هاذو`) or word-boundary choices (`قلنالك` vs `قلنا لك` counts as two wrong words). The reference transcripts are themselves inconsistent, so CER is the more faithful measure. New data needs a transcription style guide.

## How it works

```
notebooks/Preprocessing.ipynb  (local, CPU)  notebooks/hassaniya-asr-whisper-medium-fine-tuning.ipynb  (Kaggle, 2× T4)
───────────────────────────────────          ─────────────────────────────────────────────────
3 HF datasets (parquet)                      writes + runs train_whisper.py
  → 16 kHz mono (resample Dataset B)           → pre-flight check (labels, initial loss)
  → merge, drop duplicate audio (MD5)          → fine-tune, eval each epoch (val WER)
  → drop non-speech clips ([singing] …)        → keep best checkpoint, early stopping
  → normalize Arabic text                      → score test set once
  → enforce Whisper limits (30 s, 448 tok)     → results JSON, predictions, experiments.csv
  → group-aware 80/10/10 split
  → parquet (WAV bytes + text) + manifest.json ──► uploaded as a Kaggle dataset
```

**Key design choices:**
- **No leakage between splits.** Dataset B clips are grouped by source episode, and the split keeps each episode's clips in a single split. Duplicate recordings are removed before splitting; duplicates carried conflicting transcripts, so every copy is dropped. The prep notebook re-checks the saved files for overlap.
- **Audio, not features, is stored.** Log-mel features are computed during training (`Dataset.set_transform`). The dataset is ~10× smaller and works with any Whisper size.
- **Reproducible runs.** `manifest.json` records row counts and an MD5 hash per split. Every run writes its full configuration (`run_config_*.json`) and appends a row to `experiments.csv`.
- **Honest evaluation.** Checkpoints are selected on validation WER only. The test set is scored once per run.

### Decoding guards
Greedy decoding occasionally gets stuck repeating a phrase until it hits the length limit. A single looping clip adds dozens of wrong words. Generation is therefore capped at 256 tokens (the longest label is 223) and blocks any repeated 5-token sequence (`no_repeat_ngram_size=5`). These guards apply to validation (checkpoint selection) and to the test set. Each run reports how many test clips still ran away (more than 2× the reference length); it was 0 for the reported run.

### Text normalization
Applied to the training labels and to both references and predictions before scoring: Unicode NFKC, remove diacritics, fold alef variants (أ إ آ ٱ → ا) and alef maqsura (ى → ي), remove tatweel, strip Arabic and Latin punctuation, collapse whitespace. Hassaniya-specific letters such as گ are kept.

## Repository layout

| File | Purpose |
|---|---|
| `notebooks/Preprocessing.ipynb` | Data preparation, run locally. Set your folders in its **Paths** cell. |
| `train_whisper.py` | Training and evaluation script. |
| `notebooks/hassaniya-asr-whisper-medium-fine-tuning.ipynb` | Kaggle notebook that writes the script (section 3) and runs it. |
| `requirements.txt` | Python dependencies for data preparation and training. |
| `api/` | The serving side: FastAPI app, Dockerfile and its own README. |

## Reproducing

1. **Get the data:** download the three datasets and put each in its own folder (`Dataset A`, `Dataset B`, `Dataset C`):
   - A: [Mamadou-Aw/hassaniya-parler-tts](https://huggingface.co/datasets/Mamadou-Aw/hassaniya-parler-tts)
   - B: [hadramibegnoug/abc.hassaniya_ASR](https://huggingface.co/datasets/hadramibegnoug/abc.hassaniya_ASR)
   - C: [Mamadou-Aw/Hassaniya-speech-dataset](https://huggingface.co/datasets/Mamadou-Aw/Hassaniya-speech-dataset)
2. **Prepare:** `pip install -r requirements.txt`, set the folders in the **Paths** cell of `notebooks/Preprocessing.ipynb`, then run it top to bottom. It writes `hassaniya-asr-v2/` (three parquet files + `manifest.json`).
3. **Upload** that folder as a private Kaggle dataset.
4. **Train on Kaggle:** create a notebook from `notebooks/hassaniya-asr-whisper-medium-fine-tuning.ipynb`, attach the dataset, and set *Accelerator → GPU T4 ×2* and *Internet → On*. Run it with *Save & Run All*. Training takes about 1.5 hours.

The script can also be run directly:

```bash
# check labels and initial loss on a few clips (no training)
python train_whisper.py --data_dir <dataset folder> --dry_run

# train on 2 GPUs (effective batch size 8)
accelerate launch --num_processes 2 --mixed_precision fp16 train_whisper.py --data_dir <dataset folder> --grad_accum 4

# score a model on the test set only (zero-shot baseline, or a saved fine-tuned model)
python train_whisper.py --data_dir <dataset folder> --eval_only --model_id openai/whisper-medium
```

Run `python train_whisper.py --help` for all options (epochs, learning rate, batch size, early stopping patience …).

## Serve it

The fine-tuned model is available as a ready-to-run Docker image and as downloadable weights:

- Weights: [HamidAmeer/hassaniya-whisper-medium](https://huggingface.co/HamidAmeer/hassaniya-whisper-medium)
- Image: [hamidameeradil/hassaniya-asr](https://hub.docker.com/r/hamidameeradil/hassaniya-asr)

```bash
docker run --rm -p 8000:8000 hamidameeradil/hassaniya-asr:latest
```

Then open http://127.0.0.1:8000/docs and upload a clip to `POST /transcribe`. The first start downloads about 3 GB of weights. The API code, endpoints and options are described in [`api/README.md`](api/README.md).

## Implementation notes

Pitfalls when fine-tuning Whisper with the Hugging Face `Seq2SeqTrainer`, and how this code handles them:

1. **Language and task tokens in the labels.** The tokenizer is created with `language="arabic", task="transcribe"`, so labels start with `<|startoftranscript|><|ar|><|transcribe|><|notimestamps|>`, the same prefix that `generate()` uses at inference. Without it, training and inference see different prompts.
2. **Stripping the right start token.** The model adds `<|startoftranscript|>` (id 50258, `decoder_start_token_id`) itself, so the collator removes it from the labels. Comparing against `tokenizer.bos_token_id` instead does nothing, because that is `<|endoftext|>` (50257), and the model then learns to predict the start token after itself.
3. **Loss scaling with gradient accumulation.** Whisper's `forward` accepts `**kwargs`, which makes the Trainer assume the model averages loss over the whole batch and skip dividing by the accumulation steps. Setting `model.accepts_loss_kwargs = False` restores correct scaling.
4. **Checkpoint selection.** Checkpoints are chosen on validation WER, not loss. Here validation loss bottomed out three epochs before WER did (see the training curve above).
5. **Test scoring under multi-GPU training.** The test set is transcribed on one GPU, in order, so each prediction lines up with its clip.

`--dry_run` checks items 1 and 2 before any GPU time is spent: it prints the label prefix and the pretrained model's loss on a few clips.

## Limitations
- About 3 hours of audio from three scraped sources, with inconsistent transcription conventions.
- The reported numbers come from a single training run; the validation set is small, so differences of a few WER points are within noise.
- Greedy decoding only; beam search was not tried.
