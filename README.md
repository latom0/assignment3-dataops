# Assignment 3 – DataOps Incremental Pipeline

## Overview

This repository implements an automated DataOps pipeline that manages evolving time-series data using a Bronze–Silver–Gold architecture with incremental ingestion, validation, and reproducibility.

The pipeline processes a daily climate dataset and prepares an ML-ready dataset for downstream forecasting tasks.

## Key Features

* Incremental batch ingestion (5 time-ordered batches)
* Bronze–Silver–Gold data architecture
* Data lineage tracking
* Automated data validation
* Feature engineering for time-series forecasting
* Full reproducibility via DVC

## Tech Stack

* Python (Pandas, Pandera)
* DVC (Data Version Control)
* Parquet storage
* Conda environment

## Pipeline Flow

```
Batches → Bronze → Validation → Silver → Validation → Gold
```

Each new batch automatically triggers a full pipeline rebuild using `dvc repro`.

## Layers

### Bronze

* Append-only raw storage
* Lineage tracking per batch

### Silver

* Data cleaning and deduplication
* Missing value imputation
* Feature engineering (lags, rolling windows)

### Gold

* ML-ready dataset
* Next-day temperature forecasting target

## Reproducibility

Run the pipeline:

```bash
dvc repro
```

Recreate environment:

```bash
conda env create -f environment.yml
conda activate dataops
```

## Evidence

The repository includes:

* Validation logs
* Lineage metadata
* DVC pipeline configuration
* Incremental ingestion traces

## Author

Tommi Lamberg