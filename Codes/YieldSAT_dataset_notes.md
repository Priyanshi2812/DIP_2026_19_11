# Week 2 – YieldSAT Dataset Study

## Project: SAR–Optical Feature Fusion for Crop Classification

**Group:** Group B  
**Selected Countries:** Argentina and Brazil

---

## 1. What is YieldSAT?

YieldSAT is a multimodal benchmark dataset developed for high-resolution crop yield analysis. It combines crop yield measurements collected from agricultural machinery with satellite and environmental data. The dataset includes Sentinel-2 multispectral optical imagery together with yield information and additional environmental information such as weather, soil, and topography. The yield measurements are spatially processed and aligned with Sentinel-2 imagery at a 10 m × 10 m grid level. YieldSAT provides information at field and sub-field levels, making it useful for agricultural remote-sensing and machine-learning applications. For our project, YieldSAT provides the agricultural field information and optical data that will be relevant to the crop classification task.

---

## 2. What Data Does YieldSAT Provide?

YieldSAT provides several types of agricultural and remote-sensing data. These include Sentinel-2 multispectral optical imagery, crop yield measurements, weather data, soil information, and topographic information. Sentinel-2 provides optical information about agricultural fields through different spectral bands, which can be used to extract features related to crop characteristics and vegetation. Yield measurements are collected using combine harvesters equipped with yield-monitoring and GPS systems and are processed onto a 10 m spatial grid. Weather information provides environmental conditions such as temperature and precipitation, while soil and topographic information provide additional information about the growing environment. For our project, the Sentinel-2 data and crop-related information are particularly relevant, while Sentinel-1 SAR data will be obtained separately for the SAR–optical feature-fusion experiment.

---

## 3. What Countries Are Available?

YieldSAT contains agricultural field data from four countries: Argentina, Brazil, Uruguay, and Germany. The dataset covers four major crop types: corn, rapeseed, soybean, and wheat. For our project, **Group B** has been selected, consisting of **Argentina and Brazil**. The relevant crop types for these two countries are corn, soybean, and wheat. Therefore, our project will focus on agricultural fields from Argentina and Brazil and use their available crop information for the crop-classification experiment.

---

## 4. What Are Crop Labels Used For?

Crop labels identify the type of crop associated with an agricultural field, such as corn, soybean, or wheat. These labels are important for our project because crop classification is a supervised machine-learning task. The crop label acts as the ground-truth target against which the model's predicted crop class can be evaluated. For Group B, the main crop classes of interest are corn, soybean, and wheat. The satellite-derived features from Sentinel-1 and Sentinel-2 will be used as inputs to the classification model, while the corresponding crop labels will be used as the target classes.

---

## 5. What Are Yield Masks Used For?

YieldSAT processes yield measurements into a spatial grid aligned with the Sentinel-2 10 m grid. A yield mask can be used to identify pixels where valid yield information is available and distinguish them from pixels without valid yield observations. This is useful for preventing missing or invalid yield locations from being treated as actual observations. Yield information and its valid-pixel information also help in understanding the spatial agricultural regions represented in the dataset. For this project, crop labels are the primary ground-truth information for crop classification, while yield information and masks help us understand and select relevant field regions.

---

## 6. Relevance of YieldSAT to Our Project

YieldSAT is relevant to our **SAR–Optical Feature Fusion for Crop Classification** project because it provides agricultural field information and Sentinel-2 optical data that can be used to characterize crop fields. We will use Sentinel-2 information as the optical modality and acquire Sentinel-1 SAR data separately for the same study areas. The project will compare three feature settings: **optical-only features, SAR-only features, and fused optical + SAR features**. The crop labels from the relevant fields will provide the ground-truth classes for the supervised classification task. The aim is to determine how the different feature sets perform for classifying the selected crop types.

---

## 7. Group B Dataset Focus

For this project, Group B consists of **Argentina and Brazil**. The relevant crop classes are **corn, soybean, and wheat**. The dataset information for these countries will be inspected further to identify the available field identifiers, crop labels, spatial information, yield information, and Sentinel-2 data needed for subsequent processing.

---

## 8. Planned Use in the Project

The planned data workflow is:

1. Select the relevant YieldSAT fields from Argentina and Brazil.
2. Identify the crop labels associated with the fields.
3. Identify the available yield information and valid-pixel/yield-mask information.
4. Obtain or use the corresponding Sentinel-2 optical information.
5. Acquire Sentinel-1 SAR data separately for the same fields.
6. Extract features from Sentinel-1 and Sentinel-2.
7. Prepare three feature sets:
   - Optical-only
   - SAR-only
   - Optical + SAR
8. Use the crop labels as ground-truth targets for supervised classification.
9. Train and evaluate the selected machine-learning classifier.

---

## 9. Week 2 Summary

This week's work focused on understanding the YieldSAT dataset and identifying the information relevant to our project. Group B, consisting of Argentina and Brazil, was selected. The study covered the purpose of YieldSAT, the types of data it provides, the countries and crop types available, and the role of crop labels and yield masks. The next step is to inspect the actual Group B dataset structure and identify the exact field identifiers, crop-label information, yield information, spatial information, and Sentinel-2 data required for further processing.

---

## References

1. YieldSAT – Multimodal benchmark dataset for high-resolution crop yield analysis.
2. European Space Agency (ESA) – Sentinel-1 Mission.
3. European Space Agency (ESA) – Sentinel-2 Mission.
4. Google Earth Engine Data Catalog – Sentinel-1 GRD.
5. Google Earth Engine Data Catalog – Sentinel-2 MSI.
