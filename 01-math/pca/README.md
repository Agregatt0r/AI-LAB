# PCA (Principal Component Analysis)

**PCA** is a technique that finds the directions of greatest variance in data and projects
the data onto those directions to reduce dimensionality while preserving maximum information.

This folder is practice + notes for the core steps of Principal Component Analysis used
later in dimensionality reduction, data visualization, and feature extraction for ML.

Each subfolder has:
- `README.md` — theory for that topic
- `practice.py` — your exercises

## Structure

```
pca/
├── pca                         what PCA is (variance, compression, full pipeline)
├── centering                   mean of each feature and data centering
├── covariance                  covariance formula and covariance matrix
├── eigenvalues-eigenvectors    find eigenvalues and eigenvectors det(C − λI) = 0
├── principal-components        rank components by variance (PC1, PC2, explained variance)
└── projection                  project data onto components (dimensionality reduction)
```