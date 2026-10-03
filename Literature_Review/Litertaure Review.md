Orynbaikyzy et al. (2020): This study looked at crop classification using Sentinel-1 and Sentinel-2 data and compared different combinations of the two sensors. It showed that the choice of features and the availability of optical data can affect classification performance. This paper is useful for our project because we are also comparing optical-only, SAR-only, and SAR + optical approaches.

Tufail et al. (2021): The authors used Sentinel-1 and Sentinel-2 time-series data with a Random Forest classifier for crop type mapping. They compared SAR-only, optical-only, and combined data to see how each contributed to crop classification. This supports our plan to compare the three data configurations and also gives us a practical machine-learning baseline.

Eisfelder et al. (2024): This study used Sentinel-1 and Sentinel-2 time-series data together with Google Earth Engine for crop classification and agricultural monitoring. It shows how satellite data from different sources can be processed over large areas using GEE. This is relevant to our project because we also plan to use GEE for Sentinel-1 data processing.

Chen et al. (2020): This paper studied crop discrimination using Sentinel-1 SAR information along with texture features. It considered VV and VH polarization and used GLCM-based texture information to improve crop separation. The study is relevant to our project because we also plan to use VV, VH, and texture features as part of our SAR feature set.

Liu et al. (2024): This study explored the integration of optical and SAR data for crop classification using a deep-learning-based fusion approach. The work highlights how combining information from different satellite sensors can provide more useful information for distinguishing crops. It supports our decision to investigate SAR and optical data fusion, while our project will determine its actual benefit through our own experiments.

Key Findings and Recommendations
Key Findings
The research shows that Sentinel-2 is useful for spectral and vegetation information, while Sentinel-1 provides information about crop structure and moisture. VV, VH, and texture features can also help in distinguishing different crop types. Overall, the literature suggests that combining SAR and optical data may provide more useful information than using only one data source.
Recommendations
We will compare optical-only, SAR-only, and SAR + optical approaches using the same crop classes and evaluation method. Multiple observations should be used where possible, and missing or invalid data should be handled carefully. The final comparison should use measures such as accuracy, precision, recall, F1-score, and a confusion matrix to see which approach performs better in our actual experiments.

Conclusion
Overall, the Week 2 research helped us understand how Sentinel-1 and Sentinel-2 data can be useful for identifying different crops. From the papers we studied, we found that optical data gives information about vegetation, while SAR data gives additional information about crop structure and moisture. This tells us why using both types of data could be useful. Based on our research, we will compare optical-only, SAR-only, and combined SAR + optical data for corn, soybean, and wheat in Argentina and Brazil. The next step will be to implement these approaches and see how well they perform.
