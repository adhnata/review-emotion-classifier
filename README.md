# Review Emotion Classifier: from LLM labels to a fine-tuned RoBERTa

Can a small, cheap model learn to detect emotions in product reviews if an LLM does the labeling and humans only check the work?

This repo is the code for a group capstone at UNC Chapel Hill (BUSI 488 / COMP 488, Spring 2026). It builds a dataset from Amazon reviews, benchmarks six LLMs as labelers against human annotations, uses the best practical one to label about 21,000 reviews, and fine-tunes RoBERTa on those labels. The final model reaches **0.878 macro F1** on a human-labeled holdout (6 emotions plus 4 evaluation dimensions), after a first attempt at 0.600 showed the label set was too ambitious.

The use case is decision support for customer-experience teams at large e-commerce sellers: route feedback by its emotional profile (a trust violation needs a human, a product-performance gap needs engineering) instead of by star rating. The method follows the synthetic-experts approach of Ringel (2023) and the LX framework of Ludwig et al. (2026).

```
Humans define truth  ->  LLM approximates humans  ->  small model approximates the LLM
```

## Results

| | v1 (16 emotions + 4 eval dims = 20 classes) | v4 (6 emotions + 4 eval dims = 10 classes) |
|---|---|---|
| LLM labeler vs. human labels (macro F1) | 0.607 (GPT-4.1-mini, emotions) | 0.892 (GPT-4.1-mini, emotions) |
| RoBERTa baseline | 0.569 | **0.878** |
| RoBERTa, higher learning rate (5e-5) | **0.600** | 0.877 |
| RoBERTa, focal loss | n/a | 0.868 |
| RoBERTa-base (smaller model) | 0.509 | 0.865 |
| RoBERTa, 256 token context | 0.598 | n/a |

All RoBERTa scores are macro F1 on the human-labeled holdout (1,441 reviews for v1, 1,504 for v4).

Per-class F1 of the final model (RoBERTa-large baseline, v4):

| PositiveAffect | Contentment | NegativeAffect | Disappointment | Fear | Disgust |
|---|---|---|---|---|---|
| 0.95 | 0.94 | 0.87 | 0.89 | 0.74 | 0.73 |

| Trust | Commitment | Recommendation | Sentiment |
|---|---|---|---|
| 0.89 | 0.88 | 0.93 | 0.95 |

Fear and Disgust are the weak spots. They are rare (about 4% of training reviews) and often expressed indirectly. The evaluation dimensions are the most reliable part of the system.

### Read the comparison carefully

The jump from 0.600 to 0.878 is real, but it is **not** an apples-to-apples model improvement:

- **The task got easier, for reasons we measured.** Three human annotators agreed much less on the classes that were dropped or merged (Krippendorff's alpha: Guilt 0.68, Surprise 0.70, Fear 0.76, Sadness 0.78, against 0.90 overall), and the best LLM scored only 0.25 to 0.43 F1 on Pride, Sadness and Guilt. So the revision was principled, but it still means v4 answers an easier question: seven emotions (Hope, Interest, Surprise, Guilt, Pride, Sadness, Relief) are gone, and Sadness and Guilt, which scored **0.00 F1** in v1, are no longer asked about.
- **The prompt, data and context length also changed** between v1 and v4, so the gain cannot be attributed to one change.
- **Model selection used the holdout.** A validation split is carved out of the training data, but the four configs were compared on the human-labeled holdout instead. A decision-threshold sweep (0.3 to 0.5, notebook 03) was also run there: 0.3 scored best, but it is the edge of the grid, so lower thresholds were never tried. The spread between v4 configs (0.865 to 0.878) is small enough that the ranking among them should not be over-read.

## Pipeline

| Notebook | What it does |
|---|---|
| [`01_generate_datasets`](notebooks/01_generate_datasets.ipynb) | Samples 10 Amazon categories, balances ratings, builds train/test/holdout splits, tops up rare emotions using a cheap LLM screen |
| [`02_llm_benchmark_and_agreement`](notebooks/02_llm_benchmark_and_agreement.ipynb) | Labels the holdout with 6 models from 5 vendors (3 runs each), measures run-to-run reliability and agreement with humans (Krippendorff's alpha, F1, AUC, MCC), tests a routed ensemble |
| [`03_label_train_set_and_roberta_v1`](notebooks/03_label_train_set_and_roberta_v1.ipynb) | Labels 20k reviews with GPT-4.1-mini, fine-tunes RoBERTa over 20 classes in 4 configurations, sweeps the decision threshold |
| [`04_six_label_redesign_and_final_model`](notebooks/04_six_label_redesign_and_final_model.ipynb) | Merges to 6 emotions, rebuilds the holdout, re-labels, re-trains, final comparison |

The holdout was labeled by hand, independently by three annotators with a written rubric (alpha 0.90 on emotions, 0.99 on evaluation dimensions), and disagreements were reconciled by majority vote. That human labeling happens outside the notebooks.

### LLM labeler benchmark (16-label schema, 1,441 holdout reviews, majority vote of 3 runs)

| Model | Macro F1 vs. humans | Cost for the benchmark |
|---|---|---|
| Claude Haiku 4.5 | 0.624 | $5.52 |
| GPT-4.1-mini | 0.607 | $1.80 |
| GPT-4o-mini | 0.584 | $0.67 |
| DeepSeek-Chat | 0.584 | $1.26 |
| Gemini 2.5 Flash-Lite | 0.529 | $0.51 |
| Grok-3-mini | 0.485 | $0.89 |

### Design decisions worth knowing

- **Why not the most accurate labeler?** Claude Haiku 4.5 agreed best with humans, but it cost about 3 times more per review than GPT-4.1-mini and was capped at 5 parallel workers against 20, so it was also slower. GPT-4.1-mini, slightly less accurate, labeled the 17k-review training set for about $22 across three runs, roughly 25 minutes per run.
- **Run-to-run reliability matters as much as accuracy.** Gemini 2.5 Flash-Lite had a perfect self-consistency alpha of 1.00, but it returned identical labels every time while agreeing poorly with humans, so it was rigid rather than reliable. Grok-3-mini was the least stable (0.78). Claude and GPT-4.1-mini scored 0.98 and 0.95. Every labeling job uses three runs and a majority vote.
- **A routed ensemble was tried and dropped.** Picking the best model per emotion beat any single model on 12 of 16 labels (joint alpha 0.73 against 0.68 for Claude alone), but it multiplied cost and mixed models, which produces label combinations no single rater would give.
- **Imbalance handling:** weighted BCE loss and a focal-loss variant, plus targeted collection of reviews for under-represented classes.
- **Gate check:** notebook 04 refuses to continue until the human-labeled holdout is present.

## What I'd do differently

- Use the validation split I already carve out for choosing configs and thresholds (and sweep thresholds below 0.3), and keep the human holdout for one final report.
- Report confidence intervals (bootstrap over the holdout) and several training seeds. One holdout of about 1,500 reviews with 4% positive classes (Fear, Disgust) is noisy.
- Decide the label set from the human-agreement data **before** labeling 20,000 reviews, instead of after seeing v1 fail.
- The student can't beat its teacher: the classifier learns from GPT-4.1-mini labels, so 0.892 is a rough ceiling for the emotion classes. A human-labeled slice of the training data would show how much is lost.
- The ground truth is expert-inferred (annotators judged what the author felt, not what the author reported), and all data is Amazon product reviews, so generalization to other platforms is untested.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export CAPSTONE_DATA_DIR=./data          # where datasets, labels and models go (git-ignored)
export OPENAI_API_KEY=...                # needed for labeling
export ANTHROPIC_API_KEY=...             # optional, only for the benchmark in notebook 02
# also optional: GEMINI_API_KEY, XAI_API_KEY, DEEPSEEK_API_KEY

jupyter lab notebooks/
```

Run the notebooks in order. Notes:

- **Human labels are not included.** Notebook 03 expects `final_holdout.csv` and notebook 04 expects `final_holdout_v2.csv` in the data folder: the holdout reviews with 0/1 emotion columns and the evaluation dimensions, labeled by people. You will need to label your own holdout to reproduce the evaluation.
- **Labeling costs money.** Labeling the 1,504-review v4 holdout with GPT-4.1-mini costs roughly $0.65 per run. Trim the model list in notebook 02 to benchmark cheaply.
- **Notebooks 03 and 04 need a GPU** (RoBERTa-large fine-tuning). Everything also runs on Google Colab: the notebooks fall back to Colab Secrets for API keys.
- `datasets==2.17.0` is pinned because the Amazon corpus ships a loading script and the notebooks pass `trust_remote_code=True`, which executes code from the dataset repository. Read it before you run it.

## Data

Reviews come from the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) corpus (McAuley Lab, UC San Diego). Nothing from it is redistributed here. Please follow the dataset's terms of use.

## References

- Hou, Y., et al. (2024). *Bridging Language and Items for Retrieval and Recommendation.* arXiv:2403.03952. (Amazon Reviews 2023)
- Ludwig, S., et al. (2026). *Extracting consumer insight from text: A large language model approach to emotion and evaluation measurement.* [arXiv:2602.15312](https://arxiv.org/abs/2602.15312).
- Ringel, D. M. (2023). *Creating synthetic experts with generative artificial intelligence.* [SSRN 4542949](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4542949).

## Credits

This was a group capstone (a team of five). This repo covers the code and experiments, which I wrote: the data pipeline, the LLM labeling harness, and the RoBERTa training and evaluation. The team's write-up and presentation cover the rest of the project.

Several helper functions are adapted from the starter pipeline for the synthetic-experts method, marked "Ringel" in code comments.
