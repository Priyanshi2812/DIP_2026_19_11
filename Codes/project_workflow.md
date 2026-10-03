# Week 2 - Project Workflow and Feature Selection

## 1. Project Workflow

The proposed workflow for the project is:

Sentinel-2 Optical Data + Sentinel-1 SAR Data
                    ↓
             Data Preprocessing
                    ↓
            Feature Extraction
                    ↓
              Feature Fusion
                    ↓
            ML Classification
                    ↓
         Performance Evaluation
                    ↓
        Optical vs SAR vs Fusion

The three approaches to be compared are:
1. Optical-only
2. SAR-only
3. SAR + Optical feature fusion

---

## 2. Planned Optical Features

The following handcrafted features will be investigated for Sentinel-2 data:

- NDVI
- NDRE
- NDMI
- EVI
- GLCM texture features

These features are intended to capture vegetation, spectral, and texture information.

---

## 3. Planned SAR Features

The following features will be investigated for Sentinel-1 data:

- VV polarization
- VH polarization
- VV/VH ratio
- GLCM texture features

These features are intended to capture information related to crop structure, vegetation, and moisture.

---

## 4. Planned Machine Learning Pipeline

After feature extraction, three feature sets will be prepared:

- Optical feature set
- SAR feature set
- Fused optical + SAR feature set

These will be used with classical machine learning models for crop classification.

The planned evaluation metrics are:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The same evaluation procedure will be used for the three feature configurations to allow a fair comparison.

---

## 5. Next Steps

The immediate implementation plan is:

1. Finalize and obtain the required YieldSAT data for Argentina and Brazil.
2. Identify the relevant crop labels and field information.
3. Acquire corresponding Sentinel-1 SAR data using Google Earth Engine.
4. Prepare and align the Sentinel-1 and Sentinel-2 data.
5. Implement optical feature extraction.
6. Implement SAR feature extraction.
7. Construct the optical-only, SAR-only, and fused feature datasets.
8. Train and evaluate classical machine learning models.
9. Compare the performance of the three approaches.
10. Analyze the results using evaluation metrics and confusion matrices.

---

## 6. Planned Experimental Comparison

The main experimental comparison will be:

| Feature Set | Purpose |
|------------|---------|
| Optical-only | Evaluate crop classification using Sentinel-2 information |
| SAR-only | Evaluate crop classification using Sentinel-1 information |
| SAR + Optical | Determine whether combining both modalities improves classification |

The final results will be analyzed to understand the contribution of each data modality to crop classification.

---

## 7. Week 2 Outcome

This week established the preliminary methodology, feature-selection plan, experimental comparison strategy, and implementation roadmap for the project. The next phase will focus on moving from the planned methodology to actual data acquisition, preprocessing, feature extraction, and initial machine learning experiments.