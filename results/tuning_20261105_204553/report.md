# Hyperparameter tuning report

- Generated: 2026-11-05 20:45:53
- Mode: full
- Models: K-means, HDBSCAN, GMM
- Features: xl_lexeme, xlm_roberta
- Trials: 4476 (2 failed)
- Runtime: 39.0 min
- Selection metric: `combined` = 0.5 * BCubed F1 (1a) + 0.5 * Spearman rho (1b)
  - 1a: BCubed F1 on the validation split, macro-averaged over target words (fit on train)
  - 1b: Spearman rho between gold and predicted consecutive-period JSD, over all usages (refit transductively on all usages; pairs with < 10 gold usages dropped)

## Best configuration per model (validation)

| model | features | reducer | params | precision | recall | f1 | rho_1b | combined | avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HDBSCAN | xl_lexeme | none | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=True, merge_noise=True | 0.701 | 0.827 | 0.737 | 0.573 | 0.655 | 2.70 |
| GMM | xl_lexeme | AE-20 | max_components=8, covariance_type=full, reg_covar=1e-05 | 0.692 | 0.783 | 0.721 | 0.564 | 0.643 | 2.20 |
| K-means | xlm_roberta | AE-128 | max_k=6, min_silhouette=0.05, max_iter=300 | 0.684 | 0.621 | 0.633 | 0.571 | 0.602 | 2.80 |

## Best configurations on the held-out test split

Refit on train + validation, 1a scored once on test. `rho_1b` is the transductive all-usage score (1b cannot be split).

| model | features | reducer | params | val_f1 | rho_1b | test_precision | test_recall | test_f1 | test_avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HDBSCAN | xl_lexeme | none | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=True, merge_noise=True | 0.737 | 0.573 | 0.699 | 0.832 | 0.746 | 2.50 |
| GMM | xl_lexeme | AE-20 | max_components=8, covariance_type=full, reg_covar=1e-05 | 0.721 | 0.564 | 0.675 | 0.669 | 0.648 | 2.60 |
| K-means | xlm_roberta | AE-128 | max_k=6, min_silhouette=0.05, max_iter=300 | 0.633 | 0.571 | 0.648 | 0.586 | 0.574 | 2.60 |

## Baselines (validation)

| baseline | precision | recall | f1 | avg_clusters_per_word | rho_1b | combined |
| --- | --- | --- | --- | --- | --- | --- |
| One cluster per word | 0.592 | 1.000 | 0.728 | 1.00 | 0.000 | 0.364 |
| Random (up to 8 clusters) | 0.711 | 0.257 | 0.365 | 7.80 | 0.193 | 0.279 |
| TF-IDF + K-means | 0.608 | 0.933 | 0.723 | 1.60 | -0.153 | 0.285 |

## Best combined objective by model and reducer

### Features: `xl_lexeme`

| model | none | AE-20 | AE-64 | AE-128 | UMAP-10 (nn=5) | UMAP-10 (nn=15) | AE-128 + UMAP-10 | AE-20 + UMAP-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | 0.547 | 0.643 | 0.565 | 0.440 | 0.598 | 0.628 | 0.554 | 0.498 |
| HDBSCAN | 0.655 | 0.551 | 0.633 | 0.633 | 0.582 | 0.637 | 0.562 | 0.590 |
| K-means | 0.472 | 0.558 | 0.468 | 0.577 | 0.583 | 0.527 | 0.488 | 0.521 |

### Features: `xlm_roberta`

| model | none | AE-20 | AE-64 | AE-128 | UMAP-10 (nn=5) | UMAP-10 (nn=15) | AE-128 + UMAP-10 | AE-20 + UMAP-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | 0.496 | 0.636 | 0.502 | 0.522 | 0.545 | 0.537 | 0.516 | 0.542 |
| HDBSCAN | 0.551 | 0.556 | 0.599 | 0.631 | 0.531 | 0.569 | 0.546 | 0.565 |
| K-means | 0.538 | 0.574 | 0.517 | 0.602 | 0.534 | 0.461 | 0.444 | 0.485 |

## Best validation BCubed F1 (1a) by model and reducer

### Features: `xl_lexeme`

| model | none | AE-20 | AE-64 | AE-128 | UMAP-10 (nn=5) | UMAP-10 (nn=15) | AE-128 + UMAP-10 | AE-20 + UMAP-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | 0.747 | 0.736 | 0.611 | 0.623 | 0.694 | 0.730 | 0.693 | 0.689 |
| HDBSCAN | 0.764 | 0.754 | 0.762 | 0.786 | 0.697 | 0.748 | 0.658 | 0.707 |
| K-means | 0.750 | 0.724 | 0.748 | 0.734 | 0.660 | 0.681 | 0.656 | 0.656 |

### Features: `xlm_roberta`

| model | none | AE-20 | AE-64 | AE-128 | UMAP-10 (nn=5) | UMAP-10 (nn=15) | AE-128 + UMAP-10 | AE-20 + UMAP-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | 0.728 | 0.710 | 0.546 | 0.582 | 0.583 | 0.624 | 0.558 | 0.588 |
| HDBSCAN | 0.739 | 0.737 | 0.709 | 0.742 | 0.541 | 0.557 | 0.537 | 0.573 |
| K-means | 0.633 | 0.648 | 0.611 | 0.649 | 0.553 | 0.555 | 0.575 | 0.563 |

## Best Spearman rho (1b) by model and reducer

### Features: `xl_lexeme`

| model | none | AE-20 | AE-64 | AE-128 | UMAP-10 (nn=5) | UMAP-10 (nn=15) | AE-128 + UMAP-10 | AE-20 + UMAP-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | 0.491 | 0.564 | 0.537 | 0.350 | 0.503 | 0.563 | 0.454 | 0.364 |
| HDBSCAN | 0.647 | 0.467 | 0.621 | 0.570 | 0.589 | 0.529 | 0.514 | 0.526 |
| K-means | 0.195 | 0.391 | 0.187 | 0.419 | 0.508 | 0.373 | 0.321 | 0.385 |

### Features: `xlm_roberta`

| model | none | AE-20 | AE-64 | AE-128 | UMAP-10 (nn=5) | UMAP-10 (nn=15) | AE-128 + UMAP-10 | AE-20 + UMAP-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | 0.449 | 0.602 | 0.463 | 0.465 | 0.525 | 0.464 | 0.475 | 0.520 |
| HDBSCAN | 0.546 | 0.556 | 0.605 | 0.605 | 0.551 | 0.605 | 0.559 | 0.614 |
| K-means | 0.443 | 0.502 | 0.426 | 0.571 | 0.515 | 0.367 | 0.341 | 0.416 |

## Top 10 configurations per model

### HDBSCAN

| model | features | reducer | params | precision | recall | f1 | rho_1b | combined | avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HDBSCAN | xl_lexeme | none | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=True, merge_noise=True | 0.701 | 0.827 | 0.737 | 0.573 | 0.655 | 2.70 |
| HDBSCAN | xl_lexeme | none | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False, merge_noise=True | 0.701 | 0.827 | 0.737 | 0.573 | 0.655 | 2.70 |
| HDBSCAN | xl_lexeme | none | min_cluster_ratio=0.02, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False, merge_noise=True | 0.697 | 0.871 | 0.760 | 0.544 | 0.652 | 2.30 |
| HDBSCAN | xl_lexeme | none | min_cluster_ratio=0.02, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=True, merge_noise=True | 0.697 | 0.871 | 0.760 | 0.544 | 0.652 | 2.30 |
| HDBSCAN | xl_lexeme | none | min_cluster_size=3, min_samples=5, cluster_selection_method=leaf, allow_single_cluster=False, normalize_reduced=True, merge_noise=True | 0.737 | 0.581 | 0.634 | 0.647 | 0.641 | 4.00 |
| HDBSCAN | xl_lexeme | none | min_cluster_size=3, min_samples=5, cluster_selection_method=leaf, allow_single_cluster=False, normalize_reduced=False, merge_noise=True | 0.737 | 0.581 | 0.634 | 0.647 | 0.641 | 4.00 |
| HDBSCAN | xl_lexeme | none | min_cluster_size=3, min_samples=5, cluster_selection_method=leaf, allow_single_cluster=True, normalize_reduced=False, merge_noise=True | 0.737 | 0.581 | 0.634 | 0.647 | 0.641 | 4.00 |
| HDBSCAN | xl_lexeme | none | min_cluster_size=3, min_samples=5, cluster_selection_method=leaf, allow_single_cluster=True, normalize_reduced=True, merge_noise=True | 0.737 | 0.581 | 0.634 | 0.647 | 0.641 | 4.00 |
| HDBSCAN | xl_lexeme | none | min_cluster_size=10, min_samples=5, cluster_selection_method=leaf, allow_single_cluster=False, normalize_reduced=False, merge_noise=True | 0.712 | 0.788 | 0.726 | 0.551 | 0.638 | 2.30 |
| HDBSCAN | xl_lexeme | none | min_cluster_size=10, min_samples=5, cluster_selection_method=leaf, allow_single_cluster=True, normalize_reduced=True, merge_noise=True | 0.712 | 0.788 | 0.726 | 0.551 | 0.638 | 2.30 |

### GMM

| model | features | reducer | params | precision | recall | f1 | rho_1b | combined | avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | xl_lexeme | AE-20 | max_components=8, covariance_type=full, reg_covar=1e-05 | 0.692 | 0.783 | 0.721 | 0.564 | 0.643 | 2.20 |
| GMM | xlm_roberta | AE-20 | max_components=4, covariance_type=full, reg_covar=1e-05 | 0.598 | 0.912 | 0.692 | 0.581 | 0.636 | 1.30 |
| GMM | xl_lexeme | UMAP-10 (nn=15) | max_components=8, covariance_type=full, reg_covar=0.001 | 0.657 | 0.826 | 0.693 | 0.563 | 0.628 | 1.80 |
| GMM | xl_lexeme | UMAP-10 (nn=15) | max_components=4, covariance_type=full, reg_covar=0.01 | 0.635 | 0.927 | 0.730 | 0.508 | 0.619 | 1.40 |
| GMM | xl_lexeme | UMAP-10 (nn=15) | max_components=8, covariance_type=full, reg_covar=0.01 | 0.635 | 0.927 | 0.730 | 0.508 | 0.619 | 1.40 |
| GMM | xlm_roberta | AE-20 | max_components=8, covariance_type=full, reg_covar=1e-05 | 0.657 | 0.654 | 0.635 | 0.602 | 0.618 | 2.50 |
| GMM | xl_lexeme | UMAP-10 (nn=5) | max_components=4, covariance_type=full, reg_covar=0.01 | 0.687 | 0.762 | 0.694 | 0.503 | 0.598 | 2.00 |
| GMM | xl_lexeme | UMAP-10 (nn=5) | max_components=8, covariance_type=full, reg_covar=0.01 | 0.687 | 0.762 | 0.694 | 0.503 | 0.598 | 2.00 |
| GMM | xl_lexeme | UMAP-10 (nn=15) | max_components=4, covariance_type=full, reg_covar=0.001 | 0.650 | 0.830 | 0.692 | 0.465 | 0.578 | 1.70 |
| GMM | xlm_roberta | AE-20 | max_components=4, covariance_type=diag, reg_covar=0.001 | 0.700 | 0.553 | 0.598 | 0.533 | 0.565 | 3.20 |

### K-means

| model | features | reducer | params | precision | recall | f1 | rho_1b | combined | avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K-means | xlm_roberta | AE-128 | max_k=6, min_silhouette=0.05, max_iter=300 | 0.684 | 0.621 | 0.633 | 0.571 | 0.602 | 2.80 |
| K-means | xlm_roberta | AE-128 | max_k=6, min_silhouette=0.02, max_iter=100 | 0.684 | 0.621 | 0.633 | 0.571 | 0.602 | 2.80 |
| K-means | xlm_roberta | AE-128 | max_k=6, min_silhouette=0.02, max_iter=300 | 0.684 | 0.621 | 0.633 | 0.571 | 0.602 | 2.80 |
| K-means | xlm_roberta | AE-128 | max_k=6, min_silhouette=0.15, max_iter=300 | 0.684 | 0.621 | 0.633 | 0.571 | 0.602 | 2.80 |
| K-means | xlm_roberta | AE-128 | max_k=6, min_silhouette=0.05, max_iter=100 | 0.684 | 0.621 | 0.633 | 0.571 | 0.602 | 2.80 |
| K-means | xlm_roberta | AE-128 | max_k=6, min_silhouette=0.1, max_iter=100 | 0.684 | 0.621 | 0.633 | 0.571 | 0.602 | 2.80 |
| K-means | xlm_roberta | AE-128 | max_k=6, min_silhouette=0.1, max_iter=300 | 0.684 | 0.621 | 0.633 | 0.571 | 0.602 | 2.80 |
| K-means | xlm_roberta | AE-128 | max_k=6, min_silhouette=0.0, max_iter=100 | 0.684 | 0.621 | 0.633 | 0.571 | 0.602 | 2.80 |
| K-means | xlm_roberta | AE-128 | max_k=6, min_silhouette=0.0, max_iter=300 | 0.684 | 0.621 | 0.633 | 0.571 | 0.602 | 2.80 |
| K-means | xlm_roberta | AE-128 | max_k=6, min_silhouette=0.15, max_iter=100 | 0.684 | 0.621 | 0.633 | 0.571 | 0.602 | 2.80 |

## Failed configurations

| model | features | reducer | params | error |
| --- | --- | --- | --- | --- |
| GMM | xlm_roberta | AE-128 + UMAP-10 | {"max_components": 4, "covariance_type": "diag", "reg_covar": 1e-05} | ValueError: Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the number of components, increase reg_covar, or scale the input data. The numerical accuracy can also be improved by passing float64 data instead of float32. |
| GMM | xlm_roberta | AE-128 + UMAP-10 | {"max_components": 8, "covariance_type": "diag", "reg_covar": 1e-05} | ValueError: Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the number of components, increase reg_covar, or scale the input data. The numerical accuracy can also be improved by passing float64 data instead of float32. |
