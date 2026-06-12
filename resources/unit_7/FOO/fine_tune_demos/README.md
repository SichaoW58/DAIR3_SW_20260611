# Fine-tuning demos for Unit 7.3

Two end-to-end demos that match the slide deck:

| Folder | Task | Pedagogical message |
|---|---|---|
| `bibliography_reformatter/` | Messy citation -> NIH-PHS-398 format | "Fine-tuning shines for narrow, stable, deterministic tasks." |
| `nih_voice_transfer/` | Generic critique -> NIH 9-point reviewer voice | "Fine-tuning teaches a dialect that prompting cannot reliably elicit." |

Each demo contains four pieces:

1. **`assemble_dataset.py`** — turns a small seed corpus into the `train.jsonl`
   and `eval.jsonl` files expected by the OpenAI fine-tuning API. Use this as
   the template for your own dataset.
2. **`train.jsonl` / `eval.jsonl`** — small sample datasets (10-20 pairs)
   suitable for a workshop dry run. Real fine-tunes want 50-200+ pairs.
3. **`submit_finetune.py`** — uploads `train.jsonl`, kicks off the fine-tune
   job, prints the resulting `ft:...` model id when it finishes.
4. **`evaluate.py`** — runs the base model and the fine-tuned model side by
   side on `eval.jsonl`, scores both with a deterministic metric, and prints
   a comparison table.

## Prerequisites

```bash
pip install openai
export OPENAI_API_KEY="sk-..."
```

Fine-tunes are billed at training-token rates. The bundled sample datasets
(~20 pairs) cost well under \$1 to train on `gpt-4o-mini`.

## Wiring a fine-tuned model into FOO

After `submit_finetune.py` prints the `ft:...` model id, add it to
`providers.json` under the OpenAI provider:

```json
{
  "n_ordo": 200,
  "cd_model": "ft:gpt-4o-mini:your-org:nih-voice:abc123",
  "in_active": true,
  "nm_display": "GPT-4o Mini (NIH voice fine-tune)",
  "cd_provider": "openai",
  "ds_description": "Your fine-tune. Replace with actual ft:... id.",
  "cd_model_category": "chat",
  "cd_reasoning_kind": null,
  "ds_skip_sampling_fields": [],
  "ds_reasoning_ladder_json": null
}
```

The new model appears in the Model dropdown next time you launch any FOO GUI.

## License

By Juan B. Gutierrez, Professor of Mathematics, UTSA.
Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
