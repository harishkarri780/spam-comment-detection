# Model Evaluation Results

**Selected model:** Linear Support Vector Machine
**Dataset:** UCI YouTube Spam Collection (5 videos) (1956 comments)
**Evaluation:** trained on four video groups; held out Youtube05-Shakira.csv

**Deployment model:** after the test metrics were recorded, the selected pipeline was retrained on all five videos for classifying new comments.

## Holdout test metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.9189 |
| Precision Spam | 0.9390 |
| Recall Spam | 0.8851 |
| F1 Spam | 0.9112 |
| Macro F1 | 0.9183 |

## Group cross-validation (training videos)

| Model | Mean macro-F1 | Standard deviation |
|---|---:|---:|
| Multinomial Naive Bayes | 0.8292 | 0.0180 |
| Linear Support Vector Machine | 0.9242 | 0.0098 |

## Confusion matrix

Rows are actual classes; columns are predictions. Class order: ham, spam.

| Actual \ Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 186 | 10 |
| Spam | 20 | 154 |

> Results apply to this small dataset and one held-out video; they are not a guarantee of performance on other platforms or future comments.
