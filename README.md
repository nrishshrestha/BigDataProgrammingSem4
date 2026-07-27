# Bus Disruption Pattern Clustering — SCOX

Unsupervised PySpark clustering platform for identifying bus service disruption patterns from UK Bus Open Data Service (BODS) data.

## Introduction

This project is a big data analytics pipeline that ingests real UK **Bus Open Data Service (BODS)** feeds — timetables, fares, and vehicle location data — for **Stagecoach Oxfordshire (SCOX)**, and applies **unsupervised machine learning** to discover disruption patterns across the network. It combines **PySpark** for distributed data processing with **KMeans**, **DBSCAN**, and **Gaussian Mixture Model (GMM)** clustering to group bus routes and weeks by disruption frequency, severity, and duration. The result is a route-level and week-level view of where service reliability needs attention, built as a public transport analytics case study for smart city and transport planning applications.

**Keywords:** PySpark, big data pipeline, unsupervised clustering, KMeans, DBSCAN, Gaussian Mixture Model, transport analytics, Bus Open Data Service, BODS, public transit disruption, transport data science, SQLite ETL.

## 1. Data

### 1.1 Sources

| Catalogue | Source | Real / Synthetic | Records |
|---|---|---|---|
| Timetables | TransXChange XML | Real | 176,615 |
| Fares | NeTEx XML | Real | 63,428 |
| Location (AVL) | SIRI-VM, filtered to SCOX | Real | 742 |
| Disruptions | Generated | Synthetic (grounded in real line codes + BODS base rates) | 192 |

### 1.2 Why disruption data is synthetic

SCOX's Local Transport Authority does not publish disruption data on BODS. The synthetic dataset is grounded in real line codes, real stop references, and a base rate derived from live BODS statistics, rather than being arbitrary.

## 2. Pipeline

### 2.1 Notebooks (run in order)

1. `TimeTableDataPreProcessing.ipynb`
2. `FaresDataPreProcessing.ipynb`
3. `LocationDataPreProcessing.ipynb`
4. `DisruptionDataGeneration.ipynb`
5. `DatabaseCreation.ipynb` — builds `BODSDatabase.db`
6. `EDA.ipynb`
7. `DisruptionClustering.ipynb` — feature engineering + KMeans/DBSCAN/GMM
8. `ETL_Pipeline.ipynb` — complementary line-level aggregation (`line_level_summary.parquet` + `etl_line_summary` table); not yet wired into clustering

### 2.2 Architecture

```
Data Sources -> Ingestion & Preprocessing -> Storage (SQLite) -> ETL Pipeline -> Modelling -> Evaluation & Output
```

## 3. Setup

### 3.1 Requirements

```bash
pip install pyspark pandas matplotlib seaborn scikit-learn
```

### 3.2 Environment

Python 3.11, Java JDK 17.

## 4. Results

### 4.1 Model comparison

| Model | Type | Silhouette |
|---|---|---|
| KMeans (Baseline) | Partition-based | 0.356 |
| DBSCAN (Advanced) | Density-based | 0.393 |
| GMM (Advanced) | Model-based (EM) | 0.576 |

### 4.2 Interpretation

The baseline is confirmed the weakest performer; both advanced models are genuine improvements grounded in different structural assumptions rather than parameter tuning.

## 5. Notes

### 5.1 Data ethics

All real data is published by the Department for Transport under the Open Government Licence. Timetable, fares, and AVL data describe vehicles and routes, not individuals.

### 5.2 Limitations

Synthetic disruption data does not represent real historical events. The real AVL sample covers one week, not a continuous feed.
