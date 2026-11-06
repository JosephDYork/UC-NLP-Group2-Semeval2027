# Hyperparameter tuning report

- Generated: 2026-11-06 01:50:57
- Mode: full
- Models: K-means, HDBSCAN, GMM
- Features: xl_lexeme, xlm_roberta
- Trials: 4476 (0 failed)
- Runtime: 13.8 min
- Selection metric: validation BCubed F1 (Subtask 1a), macro-averaged over target words.
- Models are fit on the training split; validation gold labels are used only for selection.

## Best configuration per model (validation)

| model | features | reducer | params | precision | recall | f1 | avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False, merge_noise=False | 0.723 | 0.889 | 0.783 | 3.60 |
| GMM | xl_lexeme | UMAP-10 (nn=15) | max_components=4, covariance_type=full, reg_covar=0.01 | 0.664 | 0.956 | 0.765 | 1.50 |
| K-means | xl_lexeme | AE-64 | max_k=10, min_silhouette=0.05, max_iter=100 | 0.722 | 0.819 | 0.758 | 2.20 |

## Best configurations on the held-out test split

Refit on train + validation, then score Subtask 1a once on test.

| model | features | reducer | params | val_f1 | test_precision | test_recall | test_f1 | test_avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False, merge_noise=False | 0.783 | 0.739 | 0.805 | 0.756 | 4.70 |
| GMM | xl_lexeme | UMAP-10 (nn=15) | max_components=4, covariance_type=full, reg_covar=0.01 | 0.765 | 0.665 | 0.899 | 0.747 | 1.80 |
| K-means | xl_lexeme | AE-64 | max_k=10, min_silhouette=0.05, max_iter=100 | 0.758 | 0.715 | 0.719 | 0.689 | 2.30 |

## Baselines (held-out test)

| baseline | precision | recall | f1 | avg_clusters_per_word |
| --- | --- | --- | --- | --- |
| One cluster per word | 0.584 | 1.000 | 0.715 | 1.00 |
| Random (up to 8 clusters) | 0.692 | 0.233 | 0.335 | 7.50 |
| TF-IDF + K-means | 0.622 | 0.832 | 0.676 | 2.20 |

## Baselines (validation)

| baseline | precision | recall | f1 | avg_clusters_per_word |
| --- | --- | --- | --- | --- |
| One cluster per word | 0.592 | 1.000 | 0.728 | 1.00 |
| Random (up to 8 clusters) | 0.711 | 0.257 | 0.365 | 7.80 |
| TF-IDF + K-means | 0.608 | 0.933 | 0.723 | 1.60 |

## Best validation BCubed F1 (1a) by model and reducer

### Features: `xl_lexeme`

| model | none | AE-20 | AE-64 | AE-128 | UMAP-10 (nn=5) | UMAP-10 (nn=15) | AE-128 + UMAP-10 | AE-20 + UMAP-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | 0.747 | 0.753 | 0.620 | 0.641 | 0.695 | 0.765 | 0.687 | 0.699 |
| HDBSCAN | 0.767 | 0.742 | 0.776 | 0.783 | 0.711 | 0.753 | 0.695 | 0.704 |
| K-means | 0.750 | 0.719 | 0.758 | 0.744 | 0.665 | 0.690 | 0.691 | 0.633 |

### Features: `xlm_roberta`

| model | none | AE-20 | AE-64 | AE-128 | UMAP-10 (nn=5) | UMAP-10 (nn=15) | AE-128 + UMAP-10 | AE-20 + UMAP-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | 0.728 | 0.711 | 0.566 | 0.580 | 0.569 | 0.572 | 0.572 | 0.606 |
| HDBSCAN | 0.739 | 0.741 | 0.727 | 0.741 | 0.533 | 0.557 | 0.559 | 0.563 |
| K-means | 0.633 | 0.624 | 0.660 | 0.638 | 0.567 | 0.545 | 0.580 | 0.620 |

## Top 10 configurations per model

### HDBSCAN

| model | features | reducer | params | precision | recall | f1 | avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False, merge_noise=False | 0.723 | 0.889 | 0.783 | 3.60 |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=True, merge_noise=False | 0.723 | 0.889 | 0.783 | 3.60 |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False, merge_noise=True | 0.701 | 0.904 | 0.778 | 2.30 |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=True, merge_noise=True | 0.701 | 0.904 | 0.778 | 2.30 |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=3, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=True, merge_noise=True | 0.679 | 0.943 | 0.776 | 2.20 |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=3, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False, merge_noise=True | 0.679 | 0.943 | 0.776 | 2.20 |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=3, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=True, merge_noise=False | 0.697 | 0.932 | 0.776 | 2.90 |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=3, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False, merge_noise=False | 0.697 | 0.932 | 0.776 | 2.90 |
| HDBSCAN | xl_lexeme | AE-64 | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=True, merge_noise=False | 0.740 | 0.851 | 0.776 | 3.90 |
| HDBSCAN | xl_lexeme | AE-64 | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False, merge_noise=False | 0.740 | 0.851 | 0.776 | 3.90 |

### GMM

| model | features | reducer | params | precision | recall | f1 | avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | xl_lexeme | UMAP-10 (nn=15) | max_components=4, covariance_type=full, reg_covar=0.01 | 0.664 | 0.956 | 0.765 | 1.50 |
| GMM | xl_lexeme | UMAP-10 (nn=15) | max_components=8, covariance_type=full, reg_covar=0.01 | 0.664 | 0.956 | 0.765 | 1.50 |
| GMM | xl_lexeme | AE-20 | max_components=4, covariance_type=full, reg_covar=0.001 | 0.692 | 0.866 | 0.753 | 1.90 |
| GMM | xl_lexeme | AE-20 | max_components=8, covariance_type=full, reg_covar=0.001 | 0.692 | 0.866 | 0.753 | 1.90 |
| GMM | xl_lexeme | none | max_components=8, covariance_type=diag, reg_covar=0.001 | 0.672 | 0.897 | 0.747 | 1.70 |
| GMM | xl_lexeme | none | max_components=4, covariance_type=diag, reg_covar=0.001 | 0.672 | 0.897 | 0.747 | 1.70 |
| GMM | xl_lexeme | AE-20 | max_components=4, covariance_type=full, reg_covar=1e-05 | 0.657 | 0.884 | 0.733 | 1.70 |
| GMM | xl_lexeme | AE-20 | max_components=4, covariance_type=full, reg_covar=0.01 | 0.636 | 0.930 | 0.730 | 1.50 |
| GMM | xl_lexeme | AE-20 | max_components=8, covariance_type=full, reg_covar=0.01 | 0.636 | 0.930 | 0.730 | 1.50 |
| GMM | xl_lexeme | none | max_components=8, covariance_type=spherical, reg_covar=0.01 | 0.592 | 1.000 | 0.728 | 1.00 |

### K-means

| model | features | reducer | params | precision | recall | f1 | avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K-means | xl_lexeme | AE-64 | max_k=10, min_silhouette=0.05, max_iter=100 | 0.722 | 0.819 | 0.758 | 2.20 |
| K-means | xl_lexeme | AE-64 | max_k=10, min_silhouette=0.05, max_iter=300 | 0.722 | 0.819 | 0.758 | 2.20 |
| K-means | xl_lexeme | AE-64 | max_k=6, min_silhouette=0.0, max_iter=100 | 0.722 | 0.819 | 0.758 | 2.20 |
| K-means | xl_lexeme | AE-64 | max_k=6, min_silhouette=0.1, max_iter=100 | 0.722 | 0.819 | 0.758 | 2.20 |
| K-means | xl_lexeme | AE-64 | max_k=6, min_silhouette=0.1, max_iter=300 | 0.722 | 0.819 | 0.758 | 2.20 |
| K-means | xl_lexeme | AE-64 | max_k=6, min_silhouette=0.05, max_iter=100 | 0.722 | 0.819 | 0.758 | 2.20 |
| K-means | xl_lexeme | AE-64 | max_k=4, min_silhouette=0.1, max_iter=300 | 0.722 | 0.819 | 0.758 | 2.20 |
| K-means | xl_lexeme | AE-64 | max_k=4, min_silhouette=0.05, max_iter=100 | 0.722 | 0.819 | 0.758 | 2.20 |
| K-means | xl_lexeme | AE-64 | max_k=4, min_silhouette=0.05, max_iter=300 | 0.722 | 0.819 | 0.758 | 2.20 |
| K-means | xl_lexeme | AE-64 | max_k=4, min_silhouette=0.02, max_iter=100 | 0.722 | 0.819 | 0.758 | 2.20 |
