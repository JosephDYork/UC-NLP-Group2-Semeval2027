
# Hyperparameter tuning report

- Generated: 2026-11-25 14:24:06
- Mode: full
- Models: Correlation
- Features: xl_lexeme, xl_durel, xl_durel_span, xlm_roberta_large, mm_bert, xl_lexeme_cls, xl_durel_cls, xlm_roberta_large_cls
- Words: 60
- Rows: train 11962, validation 2563, test 2564
- Trials: 512 (0 failed)
- Runtime: 6.9 min
- Selection metric: validation BCubed F1 (Subtask 1a), macro-averaged over target words.
- Models are fit on the training split; validation gold labels are used only for selection.
- Gold senses per word: validation 4.37, test 4.32
- Confidence intervals: 95% percentile bootstrap over target words (2000 resamples). `gain_95ci` is the paired difference from the one-cluster baseline on the same words; an interval that excludes 0 means the gain holds up beyond which words were sampled.

## Best configuration per model (validation)


|model|features|reducer|params|precision|recall|f1|avg_clusters_per_word|gold_clusters_per_word|f1_95ci|gain_vs_1cluster|gain_95ci|
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|Correlation|xl_durel|none|threshold=0.575, s=10|0.652|0.781|0.696|4.42|4.37|[0.656, 0.735]|0.122|[+0.091, +0.155]|

## Best configurations on the held-out test split


Refit on train + validation, then score Subtask 1a once on test.

|model|features|reducer|params|val_f1|test_precision|test_recall|test_f1|test_avg_clusters_per_word|test_gold_clusters_per_word|test_f1_95ci|test_gain_vs_1cluster|test_gain_95ci|
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|Correlation|xl_durel|none|threshold=0.575, s=10|0.696|0.643|0.792|0.690|4.32|4.32|[0.648, 0.729]|0.108|[+0.075, +0.145]|

## Baselines (held-out test)


|baseline|precision|recall|f1|f1_95ci|avg_clusters_per_word|gold_clusters_per_word|
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|One cluster per word|0.448|1.000|0.581|[0.522, 0.634]|1.00|4.32|
|TF-IDF + K-means|0.536|0.686|0.517|[0.469, 0.564]|3.53|4.32|

## Baselines (validation)


|baseline|precision|recall|f1|f1_95ci|avg_clusters_per_word|gold_clusters_per_word|
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|One cluster per word|0.438|1.000|0.573|[0.516, 0.626]|1.00|4.37|
|TF-IDF + K-means|0.525|0.714|0.524|[0.479, 0.566]|3.10|4.37|

## Best validation BCubed F1 (1a) by model and reducer

### Features: `mm_bert`


|model|none|AE-20|AE-64|AE-128|
| :--- | :--- | :--- | :--- | :--- |
|Correlation|0.594|0.610|0.604|0.599|

### Features: `xl_durel`


|model|none|AE-20|AE-64|AE-128|
| :--- | :--- | :--- | :--- | :--- |
|Correlation|0.696|0.681|0.687|0.683|

### Features: `xl_durel_cls`


|model|none|AE-20|AE-64|AE-128|
| :--- | :--- | :--- | :--- | :--- |
|Correlation|0.692|0.683|0.686|0.687|

### Features: `xl_durel_span`


|model|none|AE-20|AE-64|AE-128|
| :--- | :--- | :--- | :--- | :--- |
|Correlation|0.694|0.676|0.679|0.683|

### Features: `xl_lexeme`


|model|none|AE-20|AE-64|AE-128|
| :--- | :--- | :--- | :--- | :--- |
|Correlation|0.672|0.664|0.669|0.666|

### Features: `xl_lexeme_cls`


|model|none|AE-20|AE-64|AE-128|
| :--- | :--- | :--- | :--- | :--- |
|Correlation|0.671|0.667|0.671|0.668|

### Features: `xlm_roberta_large`


|model|none|AE-20|AE-64|AE-128|
| :--- | :--- | :--- | :--- | :--- |
|Correlation|0.573|0.579|0.585|0.578|

### Features: `xlm_roberta_large_cls`


|model|none|AE-20|AE-64|AE-128|
| :--- | :--- | :--- | :--- | :--- |
|Correlation|0.573|0.573|0.577|0.566|

## Top 10 configurations per model

### Correlation


|model|features|reducer|params|precision|recall|f1|avg_clusters_per_word|
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|Correlation|xl_durel|none|threshold=0.575, s=10|0.652|0.781|0.696|4.42|
|Correlation|xl_durel|none|threshold=0.6, s=10|0.666|0.760|0.695|4.98|
|Correlation|xl_durel|none|threshold=0.55, s=10|0.632|0.809|0.694|3.98|
|Correlation|xl_durel_span|none|threshold=0.6, s=10|0.653|0.779|0.694|4.62|
|Correlation|xl_durel|none|threshold=0.525, s=10|0.619|0.832|0.693|3.52|
|Correlation|xl_durel|none|threshold=0.625, s=10|0.690|0.727|0.693|5.72|
|Correlation|xl_durel_cls|none|threshold=0.7, s=10|0.639|0.798|0.692|4.20|
|Correlation|xl_durel_span|none|threshold=0.55, s=10|0.618|0.825|0.688|3.55|
|Correlation|xl_durel_span|none|threshold=0.625, s=10|0.668|0.745|0.688|5.42|
|Correlation|xl_durel_cls|AE-128|threshold=0.45, s=10|0.663|0.759|0.687|4.92|
