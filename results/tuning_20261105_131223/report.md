# Hyperparameter tuning report

- Generated: 2026-11-05 13:12:23
- Mode: full
- Models: K-means, HDBSCAN, GMM
- Features: xl_lexeme, xlm_roberta
- Trials: 2684 (0 failed)
- Runtime: 11.1 min
- Selection metric: BCubed F1 on the validation split (macro-averaged over target words)

## Best configuration per model (validation)

| model | features | reducer | params | precision | recall | f1 | avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False | 0.724 | 0.902 | 0.786 | 3.50 |
| K-means | xl_lexeme | none | max_k=10, min_silhouette=0.05, max_iter=100 | 0.716 | 0.813 | 0.750 | 2.10 |
| GMM | xl_lexeme | none | max_components=8, covariance_type=diag, reg_covar=0.001 | 0.672 | 0.897 | 0.747 | 1.70 |

## Best configurations on the held-out test split

Refit on train + validation, scored once on test.

| model | features | reducer | params | val_f1 | test_precision | test_recall | test_f1 | test_avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False | 0.786 | 0.710 | 0.879 | 0.773 | 3.20 |
| K-means | xl_lexeme | none | max_k=10, min_silhouette=0.05, max_iter=100 | 0.750 | 0.727 | 0.781 | 0.744 | 2.20 |
| GMM | xl_lexeme | none | max_components=8, covariance_type=diag, reg_covar=0.001 | 0.747 | 0.696 | 0.846 | 0.747 | 2.00 |

## Baselines (validation)

| baseline | precision | recall | f1 | avg_clusters_per_word |
| --- | --- | --- | --- | --- |
| One cluster per word | 0.592 | 1.000 | 0.728 | 1.00 |
| Random (up to 8 clusters) | 0.711 | 0.257 | 0.365 | 7.80 |
| TF-IDF + K-means | 0.608 | 0.933 | 0.723 | 1.60 |

## Best validation F1 by model and reducer

### Features: `xl_lexeme`

| model | none | AE-20 | AE-64 | AE-128 | UMAP-10 (nn=5) | UMAP-10 (nn=15) | AE-128 + UMAP-10 | AE-20 + UMAP-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | 0.747 | 0.736 | 0.611 | 0.623 | 0.694 | 0.730 | 0.693 | 0.689 |
| HDBSCAN | 0.764 | 0.734 | 0.754 | 0.786 | 0.680 | 0.748 | 0.658 | 0.690 |
| K-means | 0.750 | 0.724 | 0.748 | 0.734 | 0.660 | 0.681 | 0.656 | 0.656 |

### Features: `xlm_roberta`

| model | none | AE-20 | AE-64 | AE-128 | UMAP-10 (nn=5) | UMAP-10 (nn=15) | AE-128 + UMAP-10 | AE-20 + UMAP-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | 0.728 | 0.710 | 0.546 | 0.582 | 0.583 | 0.624 | 0.558 | 0.588 |
| HDBSCAN | 0.739 | 0.737 | 0.709 | 0.742 | 0.541 | 0.556 | 0.537 | 0.573 |
| K-means | 0.633 | 0.648 | 0.611 | 0.649 | 0.553 | 0.555 | 0.575 | 0.563 |

## Top 10 configurations per model

### HDBSCAN

| model | features | reducer | params | precision | recall | f1 | avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False | 0.724 | 0.902 | 0.786 | 3.50 |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=True | 0.724 | 0.902 | 0.786 | 3.50 |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=3, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False | 0.705 | 0.910 | 0.775 | 3.20 |
| HDBSCAN | xl_lexeme | AE-128 | min_cluster_size=3, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=True | 0.705 | 0.910 | 0.775 | 3.20 |
| HDBSCAN | xl_lexeme | none | min_cluster_ratio=0.02, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=True | 0.740 | 0.844 | 0.764 | 4.80 |
| HDBSCAN | xl_lexeme | none | min_cluster_ratio=0.02, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False | 0.740 | 0.844 | 0.764 | 4.80 |
| HDBSCAN | xl_lexeme | AE-64 | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=True | 0.741 | 0.821 | 0.754 | 4.70 |
| HDBSCAN | xl_lexeme | AE-64 | min_cluster_size=5, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False | 0.741 | 0.821 | 0.754 | 4.70 |
| HDBSCAN | xl_lexeme | none | min_cluster_size=3, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=True | 0.721 | 0.860 | 0.753 | 4.30 |
| HDBSCAN | xl_lexeme | none | min_cluster_size=3, min_samples=1, cluster_selection_method=eom, allow_single_cluster=True, normalize_reduced=False | 0.721 | 0.860 | 0.753 | 4.30 |

### K-means

| model | features | reducer | params | precision | recall | f1 | avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K-means | xl_lexeme | none | max_k=10, min_silhouette=0.05, max_iter=100 | 0.716 | 0.813 | 0.750 | 2.10 |
| K-means | xl_lexeme | none | max_k=10, min_silhouette=0.15, max_iter=100 | 0.716 | 0.813 | 0.750 | 2.10 |
| K-means | xl_lexeme | none | max_k=10, min_silhouette=0.15, max_iter=300 | 0.716 | 0.813 | 0.750 | 2.10 |
| K-means | xl_lexeme | none | max_k=10, min_silhouette=0.1, max_iter=100 | 0.716 | 0.813 | 0.750 | 2.10 |
| K-means | xl_lexeme | none | max_k=10, min_silhouette=0.1, max_iter=300 | 0.716 | 0.813 | 0.750 | 2.10 |
| K-means | xl_lexeme | none | max_k=8, min_silhouette=0.0, max_iter=100 | 0.716 | 0.813 | 0.750 | 2.10 |
| K-means | xl_lexeme | none | max_k=8, min_silhouette=0.0, max_iter=300 | 0.716 | 0.813 | 0.750 | 2.10 |
| K-means | xl_lexeme | none | max_k=8, min_silhouette=0.02, max_iter=100 | 0.716 | 0.813 | 0.750 | 2.10 |
| K-means | xl_lexeme | none | max_k=8, min_silhouette=0.02, max_iter=300 | 0.716 | 0.813 | 0.750 | 2.10 |
| K-means | xl_lexeme | none | max_k=8, min_silhouette=0.1, max_iter=100 | 0.716 | 0.813 | 0.750 | 2.10 |

### GMM

| model | features | reducer | params | precision | recall | f1 | avg_clusters_per_word |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | xl_lexeme | none | max_components=8, covariance_type=diag, reg_covar=0.001 | 0.672 | 0.897 | 0.747 | 1.70 |
| GMM | xl_lexeme | none | max_components=4, covariance_type=diag, reg_covar=0.001 | 0.672 | 0.897 | 0.747 | 1.70 |
| GMM | xl_lexeme | AE-20 | max_components=8, covariance_type=full, reg_covar=0.01 | 0.633 | 0.946 | 0.736 | 1.40 |
| GMM | xl_lexeme | AE-20 | max_components=4, covariance_type=full, reg_covar=0.01 | 0.633 | 0.946 | 0.736 | 1.40 |
| GMM | xl_lexeme | AE-20 | max_components=4, covariance_type=full, reg_covar=0.001 | 0.684 | 0.855 | 0.735 | 2.00 |
| GMM | xl_lexeme | AE-20 | max_components=8, covariance_type=full, reg_covar=0.001 | 0.684 | 0.855 | 0.735 | 2.00 |
| GMM | xl_lexeme | none | max_components=4, covariance_type=spherical, reg_covar=0.001 | 0.717 | 0.776 | 0.732 | 2.30 |
| GMM | xl_lexeme | none | max_components=8, covariance_type=spherical, reg_covar=0.001 | 0.717 | 0.776 | 0.732 | 2.30 |
| GMM | xl_lexeme | UMAP-10 (nn=15) | max_components=4, covariance_type=full, reg_covar=0.01 | 0.635 | 0.927 | 0.730 | 1.40 |
| GMM | xl_lexeme | UMAP-10 (nn=15) | max_components=8, covariance_type=full, reg_covar=0.01 | 0.635 | 0.927 | 0.730 | 1.40 |
