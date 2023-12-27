# Advanced Modeling for Predicting Aviation Departure Delays

## Overview

This repository presents an in-depth analysis of flight departure delay prediction in the aviation industry. The research explores various factors contributing to delays, leveraging a massive dataset sourced from the United States Department of Transportation's Bureau of Transportation Statistics. The study employs advanced machine learning techniques, geospatial analysis, and extensive exploratory data analysis to derive valuable insights for operational improvements.

## Table of Contents

1. [Introduction](#introduction)
2. [Related Work](#related-work)
3. [Dataset Description](#dataset-description)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Effect of Airport Busyness](#effect-of-airport-busyness-on-departure-delays)
    - [Effect of Airlines Size](#effect-of-airlines-size-on-departure-delays)
    - [Correlation of Altitude with Delays](#correlation-of-altitude-with-delays)
5. [Geospatial Data Analysis](#geospatial-data-analysis)
    - [Hotspots of Flight Delays](#hotspots-of-flight-delays)
    - [Analysis of Delays by State](#analysis-of-delays-by-state)
    - [Proximity Analysis of Aviation Entities](#proximity-analysis-of-aviation-entities-and-major-airports)
6. [Methodology](#methodology)
    - [System Model](#system-model)
    - [Data Preparation](#data-preparation)
    - [Feature Selection](#feature-selection)
    - [Regression Analysis](#regression-analysis)
    - [Classification Analysis](#classification-analysis)
7. [Future Work](#future-work)
8. [Conclusion](#conclusion)
9. [References](#references)

## Introduction

Air travel has witnessed unprecedented growth, leading to an increased focus on understanding and mitigating flight departure delays. This research delves into the complexities of predicting delays, employing cutting-edge machine learning techniques to enhance accuracy and reliability.

## Related Work

The research builds upon a rich body of work, incorporating hybrid models, probabilistic approaches, and comparative analyses of machine learning models in the context of flight departure delay prediction. Notable studies include explorations of specific weather conditions and the impact of various factors on delay outcomes.

## Dataset Description

The dataset used in this research is extensive, covering flight data from January 2018 to August 2023. It encompasses a vast array of features, including detailed flight information, weather variables, and geospatial data. The sheer scale of the dataset, with millions of rows, necessitates advanced data processing techniques and analytical methodologies.

## Exploratory Data Analysis

### Effect of Airport Busyness on Departure Delays

The analysis investigates the intricate relationship between airport busyness and median departure delays. Categorizing airports based on their busyness levels, the study unveils a clear trend: departure delays increase proportionally with airport busyness. The widening spread of median departure delay values emphasizes the impact of increased congestion and operational challenges.

### Effect of Airlines Size on Departure Delays

This section explores the correlation between airline size and median departure delays. Utilizing a quantile threshold, airlines are categorized as "large" or "small." The findings reveal an inverse relationship, with larger airlines experiencing lower delays, showcasing potential operational efficiencies. The significance of strategies for improving flight punctuality is underscored.

### Correlation of Altitude with Delays

The study delves into the relationship between airport altitude and flight departure delays. Employing the K-Means algorithm for altitude clustering, the analysis reveals no clear correlation between altitude and delays. The nuanced approach includes outlier removal to ensure a more accurate representation of altitude's impact on departure delays.

## Geospatial Data Analysis

### Hotspots of Flight Delays

Geospatial analysis is employed to identify hotspots of flight delays across various geographic locations. Heatmaps visually represent median flight delays, offering insights into regions with consistently longer delays. Specific regions in the U.S., such as the East Coast, emerge as common departure delay hotspots, attributed to factors like weather conditions and airport congestion.

### Analysis of Delays by State

A heatmap is generated to provide a comprehensive overview of average median delays across different states. This analysis helps identify states with consistently higher or lower average delays, offering insights into regional variations influenced by factors such as airport infrastructure, climate, and air traffic distribution.

### Proximity Analysis of Aviation Entities and Major Airports

The impact of airspace congestion due to the proximity of aviation entities is studied. Heatmaps visualize median delays at major airports, considering the presence of Heliports, Seaplane Bases, Balloonports, and Small Airports in proximity. The correlation analysis suggests that the presence of aviation facilities may not be a significant factor causing delays at major airports.

## Methodology

### System Model

The research follows a systematic approach encompassing diverse data sources, rigorous data preprocessing, and the training of machine learning models. Evaluation metrics for regression tasks include Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), while classification tasks are assessed using F1-Score and Accuracy metrics.

### Data Preparation

Data preparation involves handling missing values, standardization, categorical encoding, and outlier removal. The treatment of missing values includes exclusion and interpolation strategies, while standardization ensures fair comparisons among numerical features. A distinctive categorical clustering algorithm is introduced for handling over 9000 unique categorical values.

### Feature Selection

To address multicollinearity, correlated features are strategically removed, enhancing the model's interpretability and generalization. Feature selection involves training a baseline linear regression model and leveraging regression coefficients to quantify feature importance. Recursive Feature Elimination (RFE) is employed to refine feature subsets for regression algorithms.

### Regression Analysis

The regression analysis employs a novel neural network architecture designed for mixed data types. Various regression models, including Linear Regression, Lasso Regression, Ridge Regression, ElasticNet Regression, Neural Network, and LightGBM Regression, are utilized. Outlier removal and RFE techniques are applied to enhance model performance, with the LightGBM Regressor emerging as the top-performing model.

### Classification Analysis

For classification tasks, the dataset is transformed, and various algorithms, including Gaussian Naive Bayes, LightGBM Classifier, Decision Tree, and Random Forest Classifier, are applied. The Random Forest Classifier outperforms other models, demonstrating the highest F1 Score and Accuracy.

## Future Work

The research lays the groundwork for future exploration and improvement in flight departure delay prediction. Potential avenues for further refinement include exploring advanced neural network architectures, optimizing clustering-based categorical encoding, deeper exploration into specific weather conditions, integration of advanced machine learning models, and collaborative efforts with aviation stakeholders for real-world validation.

## Conclusion

In conclusion, this research provides a comprehensive understanding of key factors influencing flight departure delays. The study's findings have practical implications for optimizing airline operations, resource allocation, and improving passenger experiences. The scale of the dataset and the sophisticated analytical approaches employed contribute to the robustness and significance of the research.

For detailed findings, please refer to the [full report in the repository](https://github.com/harshil-sanghvi/Advanced-Modeling-for-Predicting-Aviation-Departure-Delays/blob/main/Advanced%20Modeling%20for%20Predicting%20Aviation%20Departure%20Delays.pdf).

## References

- [U.S. Passenger Carrier Delay Costs - Airlines For America, May 2023](#)
- [Cost of delay estimates 2019 - Federal Aviation Administration, Jul 2020](#)
- [Cost of disrupted flights to the economy - AirHelp, Sep 2023](#)
- [Z. Guo, B. Yu, M. Hao, W. Wang, Y. Jiang, and F. Zong, "A novel hybrid method for flight departure delay prediction using random forest regression and maximal information coefficient," Aerospace Science and Technology, vol. 116, p. 106822, 2021](#)
- [Q. Li, R. Jing, and Z. S. Dong, "Flight delay prediction with priority information of weather and non-weather features," IEEE Transactions on Intelligent Transportation Systems, 2023](#)
- [J. G. M. Anguita and O. D. Olariaga, "Flight departure delay forecasting," Journal of Airport Management, vol. 17, no. 2, pp. 197–209, 2023](#)
- [S. Mokhtarimousavi and A. Mehrabi, "Flight delay causality: Machine learning technique in conjunction with random parameter statistical analysis," International Journal of Transportation Science and Technology, vol. 12, no. 1, pp. 230–244, 2023](#)
- [D. B. Bisandu PhD and I. Moulitsas PhD, "A deep bilstm machine learning method for flight delay prediction classification," Journal of Aviation/Aerospace Education & Research, vol. 32, no. 2, p. 4, 2023](#)
- [A. Botchkarev, "Performance metrics (error measures) in machine learning regression, forecasting and prognostics: Properties and typology," arXiv preprint arXiv:1809.03006, 2018](#)
- [ˇZ. Vujovi ́c et al., "Classification model evaluation metrics," International Journal of Advanced Computer Science and Applications, vol. 12, no. 6, pp. 599–606, 2021](#)
- [N. Fei, Y. Gao, Z. Lu, and T. Xiang, "Z-score normalization, hubness, and few-shot learning," in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 142–151, 2021](#)
- [V. Barnett, T. Lewis, et al., Outliers in statistical data, vol. 3. Wiley New York, 1994](#)
