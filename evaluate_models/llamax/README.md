# LLaMAX2 rescoring

Rescores the Llama-2 sentence pairs with LLaMAX2, so it gets the same inputs in all five languages. Run from the repo root.

```
python -m evaluate_models.llamax.main --input results/all_sentence_scores.csv --output results/llamax.csv
```
