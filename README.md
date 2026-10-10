# SAR-Optical Feature Fusion for Crop Type Classification

**Multimodal Remote Sensing | Digital Image Processing | Agricultural Computer Vision**

## Overview

Crop type classification using satellite imagery can support agricultural monitoring and large-scale land-use analysis. However, optical imagery and Synthetic Aperture Radar (SAR) capture different characteristics of the Earth's surface.

This project explores the integration of Sentinel-2 optical features and Sentinel-1 SAR backscatter features to investigate their potential for distinguishing major crop types, including corn, soybean, and wheat.

The goal is to develop a reproducible processing pipeline that extracts, analyses, and combines complementary satellite-derived features for crop classification.

## Objectives

- Process and analyse Sentinel-2 optical imagery.
- Calculate vegetation indices such as the Normalized Difference Vegetation Index (NDVI).
- Inspect Sentinel-1 VV and VH polarization bands.
- Develop a consistent feature extraction and integration pipeline.
- Investigate the contribution of optical and SAR features to crop type classification.
- Evaluate classification performance using suitable machine learning methods once the fused dataset is available.

## Methodology

```text
YieldSAT Samples and Crop Labels
              |
      Data Preprocessing
              |
       +------+------+
       |             |
 Sentinel-2      Sentinel-1
 Optical Data    SAR Data
       |             |
   NDVI and       VV and VH
   Spectral       Backscatter
   Features       Features
       |             |
       +------+------+
              |
      Feature Integration
              |
      Crop Classification
              |
     Model Evaluation
```

## Preliminary Results

The current analysis uses 100 optical sample records from Argentina and Brazil.

| Metric | Preliminary result |
|---|---:|
| Total optical records | 100 |
| Corn samples | 34 |
| Soybean samples | 34 |
| Wheat samples | 32 |
| Mean NDVI | 0.3566 |
| Minimum NDVI | 0.1884 |
| Maximum NDVI | 0.6763 |

NDVI was calculated from the Sentinel-2 red (B04) and near-infrared (B08) bands and verified against the supplied values. Exploratory visualizations were generated to examine NDVI distributions, crop-wise variation, and spectral band distributions.

The Sentinel-1 component has been explored using a two-band VV/VH test raster. Field-level SAR-optical matching and the final fused feature table remain part of the ongoing implementation.

## Technology Stack

- **Language:** Python
- **Data processing:** NumPy, Pandas
- **Geospatial raster processing:** Rasterio
- **Visualisation:** Matplotlib
- **Satellite data access:** Google Earth Engine
- **Development environment:** Google Colab and Jupyter Notebook
- **Planned classification:** Scikit-learn

## Repository Structure

```text
.
├── Codes/
│   └── SAR_Optical_Integrated_Pipeline.ipynb
├── Results/
│   ├── Week3_crop_counts.csv
│   ├── Week3_optical_features_verified.csv
│   └── Week3_data_quality_summary.csv
├── Reports/
├── Literature_Review/
├── Mid_Sem_Report/
├── End_Sem_Report/
└── README.md
```

## Getting Started

1. Clone or download this repository.
2. Open the notebook in Google Colab or Jupyter Notebook.
3. Install the required Python libraries if they are not already available.
4. Provide the input CSV files and SAR test raster referenced by the notebook.
5. Run the notebook cells in sequence to reproduce the current optical analysis, visualizations, and SAR data-quality checks.

The full YieldSAT dataset is not included in this repository because of its large size. Access to the original data and its associated metadata is required for reproducing the complete workflow.

## Current Scope and Next Steps

The current implementation establishes the preliminary optical analysis and SAR raster inspection components. The next stage is to validate the spatial metadata, obtain matching Sentinel-1 features for the optical samples, and construct a fused feature table.

Further work will focus on additional spectral and radar features, feature-level analysis, and machine learning experiments comparing optical-only, SAR-only, and combined inputs.

## Data and Reproducibility

The project uses YieldSAT-derived sample records and Sentinel satellite data. Dataset access, licensing, and attribution requirements should be followed when reusing or redistributing the data.

The repository is intended to maintain the processing code, derived outputs, and documentation needed to track and reproduce the analysis.

## Author

Developed as a Digital Image Processing project focused on multimodal satellite imagery and agricultural crop classification.
