# SAR–Optical Feature Fusion for Crop Classification

## Project Overview

This project investigates the use of **Synthetic Aperture Radar (SAR)** and **optical satellite imagery** for crop classification.

The project focuses on combining features obtained from **Sentinel-1 SAR data** and **Sentinel-2 optical data** to improve the representation of agricultural areas. The combined SAR–optical features can then be used as input to a machine learning classification model for identifying different crop classes.

The analysis is carried out using **Google Earth Engine (GEE)**.

---

## Objectives

The main objectives of this project are:

1. To understand the fundamentals of Synthetic Aperture Radar (SAR).
2. To study **Sentinel-1 SAR GRD** data available in Google Earth Engine.
3. To understand the importance of Sentinel-1 **VV and VH polarization** features.
4. To study Sentinel-2 optical spectral information relevant to agriculture.
5. To extract useful features from both SAR and optical imagery.
6. To combine SAR and optical features through feature fusion.
7. To investigate the usefulness of the fused features for crop classification.
8. To develop a workflow that can be implemented using Google Earth Engine and machine learning.

---

## What is SAR?

**Synthetic Aperture Radar (SAR)** is a remote-sensing technology that uses radar signals to observe the Earth's surface.

Unlike optical sensors, which depend mainly on reflected sunlight, SAR actively sends microwave signals toward the Earth's surface and measures the signal that is returned to the satellite.

### Optical Remote Sensing

```text
Sunlight
   ↓
Earth surface / crop
   ↓
Reflected light
   ↓
Satellite sensor
```

Optical sensors therefore depend on the availability of sunlight and can be affected by cloud cover.

### SAR Remote Sensing

```text
Satellite
    ↓
Radar signal
    ↓
Earth surface / crop
    ↓
Returned radar signal
    ↑
Satellite
```

Because SAR actively generates its own microwave signal, it can acquire observations during both day and night and is less affected by cloud cover.

---

## Why Use SAR for Crop Classification?

Different crops can interact differently with microwave signals because of differences in:

* Crop structure
* Plant height
* Biomass
* Surface roughness
* Moisture
* Vegetation arrangement

These differences can affect the radar signal returned to the satellite.

Therefore, SAR measurements can provide information about agricultural fields that may complement information obtained from optical imagery.

---

## Sentinel-1

**Sentinel-1** is a European Earth-observation mission providing radar imagery.

For this project, Sentinel-1 **Ground Range Detected (GRD)** data are used through Google Earth Engine.

The relevant Google Earth Engine collection is:

```text
COPERNICUS/S1_GRD
```

Sentinel-1 provides measurements using different polarization configurations. Two important polarization channels used in this project are:

* **VV**
* **VH**

---

## VV and VH Polarization

### VV

**VV** represents a vertically transmitted and vertically received radar signal.

It can provide information related to the interaction of the radar signal with the surface and vegetation.

### VH

**VH** represents a vertically transmitted and horizontally received radar signal.

VH is particularly useful for characterizing vegetation structure because the radar signal can undergo scattering interactions within vegetation.

### VV + VH

Using both VV and VH provides complementary information about the observed agricultural area.

The exact usefulness of each feature depends on factors such as:

* Crop type
* Crop growth stage
* Field structure
* Moisture
* Acquisition conditions

---

# Sentinel-2 Optical Data

Sentinel-2 is an optical Earth-observation mission that provides multispectral imagery.

Optical bands provide information about the spectral response of Earth's surface.

For agricultural applications, optical information can be particularly useful for describing vegetation characteristics.

Examples of useful optical information include:

* Visible spectral bands
* Near-infrared information
* Short-wave infrared information
* Vegetation indices

---

## SAR vs Optical Data

| Property                   | Optical                | SAR                             |
| -------------------------- | ---------------------- | ------------------------------- |
| Energy source              | Sunlight               | Active radar signal             |
| Day/night operation        | Mainly daytime         | Day and night                   |
| Cloud sensitivity          | Affected by clouds     | Can acquire through clouds      |
| Information                | Spectral response      | Radar backscatter               |
| Important crop information | Vegetation reflectance | Structure, moisture, scattering |
| Example satellite          | Sentinel-2             | Sentinel-1                      |

The two sensor types provide different and complementary information.

This motivates the use of **SAR–optical feature fusion**.

---

# SAR–Optical Feature Fusion

Feature fusion means combining information from different sources into a common feature set.

In this project, features from Sentinel-1 and Sentinel-2 are combined.

Conceptually:

```text
                 Satellite Data
                       |
             -----------------------
             |                     |
        Sentinel-1             Sentinel-2
           SAR                    Optical
             |                     |
          VV, VH          Spectral bands/indices
             |                     |
             -------- Feature --------
                     Fusion
                       |
                Combined Dataset
                       |
              Machine Learning
                       |
                Crop Classification
```

The purpose of feature fusion is to provide the classification model with complementary information from both sensing technologies.

---

# Project Workflow

The overall workflow is:

```text
Define Study Area
       ↓
Acquire Sentinel-1 SAR Data
       ↓
Select VV and VH Features
       ↓
Acquire Sentinel-2 Optical Data
       ↓
Select Relevant Optical Features
       ↓
Preprocess Satellite Data
       ↓
Generate SAR and Optical Features
       ↓
Fuse Features
       ↓
Prepare Training Samples
       ↓
Train Machine Learning Classifier
       ↓
Classify Crop Types
       ↓
Evaluate Classification
```

---

# Google Earth Engine

Google Earth Engine (GEE) is used as the main platform for accessing and processing satellite imagery.

It provides access to large collections of Earth-observation data and allows satellite images to be processed using cloud-based computation.

For this project, GEE is used for:

* Accessing Sentinel-1 imagery
* Accessing Sentinel-2 imagery
* Filtering imagery by location and time
* Selecting relevant bands
* Creating features
* Combining SAR and optical information
* Preparing data for classification

---

# Sentinel-1 Data Collection

The Sentinel-1 GRD collection used in Google Earth Engine is:

```javascript
COPERNICUS/S1_GRD
```

A typical workflow begins by filtering the collection according to the study requirements.

Example structure:

```javascript
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(studyArea)
  .filterDate(startDate, endDate);
```

The collection can then be filtered according to relevant acquisition characteristics and polarization.

For example:

```javascript
var s1VV = s1
  .filter(ee.Filter.listContains(
    'transmitterReceiverPolarisation', 'VV'
  ));

var s1VH = s1
  .filter(ee.Filter.listContains(
    'transmitterReceiverPolarisation', 'VH'
  ));
```

The exact filters used should match the final study area, time period, and experiment.

---

# Sentinel-2 Data

Sentinel-2 imagery can similarly be filtered by:

* Study area
* Observation period
* Cloud conditions

Relevant spectral bands and vegetation indices can then be extracted.

A simplified workflow is:

```text
Sentinel-2 Collection
        ↓
Filter by Study Area
        ↓
Filter by Date
        ↓
Cloud Filtering
        ↓
Select Relevant Bands
        ↓
Calculate Indices
        ↓
Create Optical Features
```

---

# Feature Preparation

The feature dataset consists of information derived from both sensor types.

### SAR Features

The main SAR features considered are:

```text
VV
VH
```

### Optical Features

Optical features can include selected Sentinel-2 spectral bands and derived vegetation indices.

The final feature set therefore has the general form:

```text
Features =
[SAR features + Optical features]
```

For example:

```text
[VV, VH, Optical Band 1, Optical Band 2, ..., Vegetation Index]
```

The exact final feature list depends on the experiment and available data.

---

# Machine Learning Classification

After feature extraction and fusion, the resulting dataset can be used for supervised machine learning.

The basic structure is:

```text
Fused Features
      +
Crop Labels
      ↓
Training Dataset
      ↓
Machine Learning Model
      ↓
Predicted Crop Classes
```

The model learns the relationship between satellite-derived features and known crop classes.

The trained model can then be applied to unseen pixels or field samples to produce crop-class predictions.

---

# Classification Evaluation

The classification results should be evaluated using appropriate classification metrics.

Possible evaluation measures include:

* Overall Accuracy
* Confusion Matrix
* Producer's Accuracy
* User's Accuracy
* F1-score

The evaluation should be performed using appropriate validation or test samples rather than using the same samples used for training.

---

# Why Combine SAR and Optical Data?

SAR and optical sensors observe the Earth's surface using different physical mechanisms.

Optical imagery provides information related to the spectral response of vegetation and the surface.

SAR provides information related to microwave backscatter and interactions with surface and vegetation structure.

Therefore:

```text
Optical information
        +
SAR information
        ↓
Complementary features
        ↓
Richer representation of agricultural fields
```

The project investigates whether this combined representation can be useful for crop classification.

---

# Expected Outcome

The project aims to develop a complete workflow for:

1. Obtaining Sentinel-1 SAR data.
2. Extracting VV and VH features.
3. Obtaining Sentinel-2 optical information.
4. Extracting relevant optical features.
5. Combining SAR and optical features.
6. Training a machine learning classification model.
7. Producing crop-class predictions.
8. Evaluating the resulting classification.

The final experimental results will determine how useful the fused feature set is for the selected study area and crop classes.

---

# Limitations

Several factors can influence SAR–optical crop classification:

* Cloud contamination in optical imagery
* Differences in acquisition dates between sensors
* Crop growth stage
* Soil moisture
* Field size and boundaries
* Similar spectral responses between crop types
* Similar radar backscatter between different crops
* Quality and quantity of training samples
* Seasonal variations

Therefore, the classification results should be interpreted in the context of the selected study area, observation period, features, and training data.

---

# Project Structure

A possible project repository structure is:

```text
SAR-Optical-Crop-Classification/
│
├── README.md
│
├── gee/
│   ├── sentinel1_processing.js
│   ├── sentinel2_processing.js
│   └── feature_fusion.js
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── classification.ipynb
│
├── results/
│   ├── maps/
│   ├── figures/
│   └── metrics/
│
└── docs/
    └── project_report.md
```

---

# Technologies and Tools

The project uses:

* **Google Earth Engine**
* **Sentinel-1 SAR**
* **Sentinel-2 optical imagery**
* **JavaScript for GEE processing**
* **Machine Learning**
* **Remote Sensing**
* **Geospatial Data Analysis**

Additional Python-based tools may be used for machine learning and analysis depending on the final implementation.

---

# Key Terms

| Term                | Meaning                                                |
| ------------------- | ------------------------------------------------------ |
| SAR                 | Synthetic Aperture Radar                               |
| GEE                 | Google Earth Engine                                    |
| GRD                 | Ground Range Detected                                  |
| VV                  | Vertical transmit, Vertical receive polarization       |
| VH                  | Vertical transmit, Horizontal receive polarization     |
| SAR–Optical Fusion  | Combining SAR and optical features                     |
| Crop Classification | Assigning agricultural areas to crop classes           |
| Backscatter         | Returned radar signal measured by SAR                  |
| Feature             | A measurable variable used by the classification model |

---

# Conclusion

This project explores **SAR–optical feature fusion for crop classification** using Sentinel-1 and Sentinel-2 satellite data.

Sentinel-1 contributes radar-based information through features such as VV and VH, while Sentinel-2 contributes optical spectral information and potentially derived vegetation indices.

By combining these complementary sources of information, the project develops a workflow for preparing satellite-derived features for machine learning-based crop classification.

The final performance of the approach will be assessed using the classification results and appropriate evaluation metrics.

---

# References

1. European Space Agency (ESA) — Sentinel-1 Mission.
2. European Space Agency (ESA) — Sentinel-2 Mission.
3. Google Earth Engine Data Catalog — Sentinel-1 SAR GRD.
4. Google Earth Engine Data Catalog — Sentinel-2 MSI.
5. Google Earth Engine Documentation — Image Collections and Classification.

---

## Project Topic

**SAR–Optical Feature Fusion for Crop Classification using Sentinel-1 SAR and Sentinel-2 Optical Features**

## Main Dataset

```text
Sentinel-1 GRD
Google Earth Engine Collection:
COPERNICUS/S1_GRD
```

## Main SAR Features

```text
VV
VH
```

## Main Idea

```text
Sentinel-1 SAR
      +
Sentinel-2 Optical
      ↓
Feature Fusion
      ↓
Machine Learning
      ↓
Crop Classification
```