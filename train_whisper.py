"""Fine-tune Whisper on the Hassaniya ASR dataset produced by Preprocessing.ipynb.

Single-GPU:  python train_whisper.py --data_dir DIR
Two GPUs:    accelerate launch --num_processes 2 --mixed_precision fp16 train_whisper.py --grad_accum 4
Pre-flight:  python train_whisper.py --dry_run   (checks labels + loss on a few batches, no training)
Resume:      same command plus --resume   (continues from the newest checkpoint in --output_dir)
Test only:   python train_whisper.py --eval_only --model_id <saved model folder or hub id>
             (e.g. the zero-shot baseline: --model_id openai/whisper-medium)

Checkpoints are selected on validation WER; the test set is scored once, at the end.
"""

import argparse
import csv
import glob
import io
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Dataset
from jiwer import cer, wer
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
import transformers

SAMPLING_RATE = 16000
RUNAWAY_LENGTH_RATIO = 2.0  # a prediction with more than 2x the reference's words counts as a runaway
LANGUAGE = "arabic"
TASK = "transcribe"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="openai/whisper-medium")
    p.add_argument("--data_dir", default=None,
                   help="Folder with manifest.json + parquet files; auto-found under /kaggle/input if omitted")
    p.add_argument("--output_dir", default="/kaggle/working/whisper-medium-hassaniya")
    p.add_argument("--run_name", default=None)
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--early_stopping_patience", type=int, default=2)
    # Longest label in dataset v2 is 223 tokens; a tighter cap limits the damage of a looping generation
    p.add_argument("--generation_max_length", type=int, default=256)
    p.add_argument("--no_repeat_ngram_size", type=int, default=5,
                   help="Block any repeated n-token sequence during generation (0 = off); stops looping output")
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--eval_only", action="store_true",
                   help="Score --model_id (hub id or a saved folder such as <output_dir>/best) on the test set; no training")
    return p.parse_args()


# Identical copy of normalize_arabic in Preprocessing.ipynb - keep the two in sync.
def normalize_arabic(text):
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))

    # Remove Arabic diacritics
    text = re.sub(r"[ؐ-ًؚ-ٰٟۖ-ۭ]", "", text)

    # Normalize alef variants
    text = re.sub(r"[أإآٱ]", "ا", text)

    # Alef maqsura -> ya
    text = text.replace("ى", "ي")

    # Remove tatweel
    text = text.replace("ـ", "")

    # Remove punctuation (Arabic ، ؛ ؟ included; letters, digits and gaf stay)
    text = re.sub(r"[^\w\s]|_", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def find_data_dir(data_dir):
    if data_dir:
        return data_dir
    manifests = glob.glob("/kaggle/input/**/manifest.json", recursive=True)
    if len(manifests) != 1:
        raise SystemExit(f"Pass --data_dir; found {len(manifests)} manifest.json files: {manifests}")
    return os.path.dirname(manifests[0])


def load_split(data_dir, manifest, split):
    path = os.path.join(data_dir, manifest["splits"][split]["file"])
    ds = Dataset.from_parquet(path)
    expected = manifest["splits"][split]["rows"]
    if len(ds) != expected:
        raise SystemExit(f"{split}: {len(ds)} rows on disk, manifest says {expected}")
    return ds


class WhisperTransform:
    """Decode WAV bytes -> log-mel features, text -> labels with the full Whisper prefix.

    A module-level class (not a closure) so dataloader workers can pickle it on Windows too.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        features, labels = [], []
        for audio, text in zip(batch["audio"], batch["text"]):
            waveform, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            if sr != SAMPLING_RATE or waveform.ndim != 1:
                raise ValueError(f"Expected 16 kHz mono, got sr={sr} shape={waveform.shape}")
            features.append(
                self.processor.feature_extractor(waveform, sampling_rate=SAMPLING_RATE).input_features[0]
            )
            labels.append(self.processor.tokenizer(text).input_ids)
        return {"input_features": features, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        # The model prepends <|startoftranscript|> itself when it shifts labels right.
        # Compare against decoder_start_token_id (50258), not tokenizer.bos_token_id (50257).
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def make_compute_metrics(processor):
    tokenizer = processor.tokenizer

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = np.where(pred.label_ids == -100, tokenizer.pad_token_id, pred.label_ids)
        pred_ids = np.where(pred_ids == -100, tokenizer.pad_token_id, pred_ids)

        # Labels were normalized at prep time; normalizing again is a no-op for refs
        preds = [normalize_arabic(s) for s in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
        refs = [normalize_arabic(s) for s in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

        return {
            "wer": 100 * wer(refs, preds),
            "cer": 100 * cer(refs, preds),
        }

    return compute_metrics


def environment_info():
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "gpus": gpus,
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def dry_run(model, processor, collator, train_ds):
    """One-process check that labels carry the right prefix and the loss is sane before a long run."""
    sample = train_ds[0]
    print("Label prefix:", processor.tokenizer.convert_ids_to_tokens(sample["labels"][:4]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    batch = collator([train_ds[i] for i in range(4)])
    print("Collated label start:", processor.tokenizer.convert_ids_to_tokens(batch["labels"][0][:3].tolist()))

    with torch.no_grad():
        out = model(
            input_features=batch["input_features"].to(device),
            labels=batch["labels"].to(device),
        )
    print(f"Pretrained loss on 4 samples: {out.loss.item():.3f}  (expect roughly 1-6; far higher means malformed labels)")

    generated = model.generate(batch["input_features"][:2].to(device), language=LANGUAGE, task=TASK, max_new_tokens=60)
    for ids in generated:
        print("Zero-shot:", processor.tokenizer.decode(ids, skip_special_tokens=True))


def main():
    args = parse_args()
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    data_dir = find_data_dir(args.data_dir)

    with open(os.path.join(data_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    processor = WhisperProcessor.from_pretrained(args.model_id, language=LANGUAGE, task=TASK)

    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    # Whisper's forward takes **kwargs, so Trainer assumes it averages loss over num_items_in_batch
    # and skips the grad-accum division; it doesn't (plain mean CE), so loss would be inflated by grad_accum x GPUs.
    model.accepts_loss_kwargs = False
    # Applies to validation generation (Trainer) and test scoring alike
    model.generation_config.no_repeat_ngram_size = args.no_repeat_ngram_size

    train_ds = load_split(data_dir, manifest, "train")
    val_ds = load_split(data_dir, manifest, "validation")
    test_ds = load_split(data_dir, manifest, "test")
    # Materialize now: datasets>=4 returns a lazy Column that would go through the transform later
    test_sources = list(test_ds["source"])
    test_texts = list(test_ds["text"])
    for ds in (train_ds, val_ds, test_ds):
        ds.set_transform(WhisperTransform(processor))

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    if args.dry_run:
        print(f"Dataset {manifest['dataset_version']} from {data_dir}: "
              f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
        dry_run(model, processor, collator, train_ds)
        return

    if args.eval_only:
        predictions, metrics = score_test(model, processor, collator, test_ds, test_texts, test_sources, args)
        write_results(args, run_name, manifest, predictions, metrics, training_info={})
        return

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,

        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.num_beams,

        logging_strategy="steps",
        logging_steps=25,

        remove_unused_columns=False,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=make_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    config = {
        "run_name": run_name,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": vars(args),
        "data_dir": data_dir,
        "dataset_version": manifest["dataset_version"],
        "dataset_splits": {k: {"rows": v["rows"], "md5": v["md5"]} for k, v in manifest["splits"].items()},
        "effective_batch_size": args.batch_size * args.grad_accum * int(os.environ.get("WORLD_SIZE", 1)),
        "environment": environment_info(),
    }
    if trainer.is_world_process_zero():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"run_config_{run_name}.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    trainer.train(resume_from_checkpoint=True if args.resume else None)

    best_dir = os.path.join(args.output_dir, "best")
    trainer.save_model(best_dir)
    if not trainer.is_world_process_zero():
        return
    processor.save_pretrained(best_dir)

    # Score the test set once, with the checkpoint chosen on validation WER.
    # Done on rank 0 only, in dataset order, so every prediction lines up with its test clip
    # (outputs gathered from trainer.predict across GPUs are not guaranteed to match the test set row for row).
    predictions, metrics = score_test(
        trainer.accelerator.unwrap_model(trainer.model), processor, collator, test_ds, test_texts, test_sources, args
    )

    state = trainer.state
    best_val = next(
        (log for log in state.log_history if log.get("eval_wer") == state.best_metric), {}
    )
    training_info = {
        "best_checkpoint": state.best_model_checkpoint,
        "epochs_trained": round(state.epoch, 2),
        "effective_batch_size": config["effective_batch_size"],
        "val_wer": best_val.get("eval_wer"),
        "val_cer": best_val.get("eval_cer"),
    }
    write_results(args, run_name, manifest, predictions, metrics, training_info)


def score_test(model, processor, collator, test_ds, test_texts, test_sources, args):
    """Generate test transcripts on a single device, in dataset order, and score them."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    preds = []
    for start in range(0, len(test_ds), args.eval_batch_size):
        batch = collator([test_ds[i] for i in range(start, min(start + args.eval_batch_size, len(test_ds)))])
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            ids = model.generate(
                batch["input_features"].to(device),
                language=LANGUAGE,
                task=TASK,
                max_length=args.generation_max_length,
                num_beams=args.num_beams,
            )
        preds.extend(normalize_arabic(s) for s in processor.tokenizer.batch_decode(ids, skip_special_tokens=True))
        print(f"test {min(start + args.eval_batch_size, len(test_ds))}/{len(test_ds)}", flush=True)

    refs = [normalize_arabic(t) for t in test_texts]
    predictions = pd.DataFrame({"source": test_sources, "reference": refs, "prediction": preds})

    # Runaway generations (looping output) inflate WER on a few clips; count them so runs can be compared
    length_ratio = predictions["prediction"].str.split().str.len() / predictions["reference"].str.split().str.len()
    runaway = predictions[length_ratio > RUNAWAY_LENGTH_RATIO]
    kept = predictions[length_ratio <= RUNAWAY_LENGTH_RATIO]

    metrics = {
        "test_wer": 100 * wer(refs, preds),
        "test_cer": 100 * cer(refs, preds),
        "test_per_source": {
            source: {"rows": len(g), "wer": 100 * wer(list(g["reference"]), list(g["prediction"])),
                     "cer": 100 * cer(list(g["reference"]), list(g["prediction"]))}
            for source, g in predictions.groupby("source")
        },
        "test_runaway_clips": len(runaway),
        "test_runaway_per_source": runaway["source"].value_counts().to_dict(),
        "test_wer_without_runaway": 100 * wer(list(kept["reference"]), list(kept["prediction"])),
    }
    return predictions, metrics


def write_results(args, run_name, manifest, predictions, metrics, training_info):
    os.makedirs(args.output_dir, exist_ok=True)
    predictions.to_csv(os.path.join(args.output_dir, f"test_predictions_{run_name}.csv"), index=False, encoding="utf-8")

    results = {
        "run_name": run_name,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": args.model_id,
        "eval_only": args.eval_only,
        **training_info,
        **metrics,
    }
    with open(os.path.join(args.output_dir, f"results_{run_name}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))

    log_path = os.path.join(args.output_dir, "experiments.csv")
    row = {
        "run_name": run_name,
        "model_id": args.model_id,
        "dataset_version": manifest["dataset_version"],
        "gpus": "+".join(environment_info()["gpus"]),
        "effective_batch_size": training_info.get("effective_batch_size"),
        "lr": None if args.eval_only else args.lr,
        "epochs_trained": training_info.get("epochs_trained"),
        "best_checkpoint": training_info.get("best_checkpoint"),
        "val_wer": training_info.get("val_wer"),
        "test_wer": metrics["test_wer"],
        "test_cer": metrics["test_cer"],
        "test_runaway_clips": metrics["test_runaway_clips"],
        "generation_max_length": args.generation_max_length,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
