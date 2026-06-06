# Kaggle Competition: Jane Street Market Prediction

This repository contains my complete solution for the "Jane Street Market Prediction" Kaggle competition. The goal was to develop a model to predict in real-time whether a financial opportunity would yield a positive return. My final solution is a sophisticated, multi-layered ensemble of diverse models, which proved to be highly effective in this challenging time-series forecasting task.

## Project Overview & Methodology

The core of my strategy was to build a powerful and diverse ensemble system rather than relying on a single model. This approach mitigates the risk of individual model weaknesses and captures a wider range of patterns in the noisy financial data. The workflow was as follows:

1.  **Feature Engineering:**
    * Processed the dataset's 130 anonymized market data features.
    * Handled missing values using a forward-fill strategy appropriate for time-series data and created new interaction features to enhance signal.

2.  **Diverse Model Training:**
    I developed and trained four distinct types of models to ensure a high degree of diversity in the ensemble:
    * **Gradient Boosted Trees (XGBoost):** A powerful tree-based model to capture complex, non-linear interactions.
    * **Robust Linear Model (Ridge Regression):** A simple and efficient linear model that serves as a stable baseline and adds diversity to the ensemble.
    * **Standard Deep Learning Model (MLP):** A Multi-Layer Perceptron with a ResNet-like architecture, designed to learn deep feature representations.
    * **Advanced Deep Learning Model (TabM):** I implemented and trained an advanced model for tabular data based on the research paper "TabM". This specialized architecture is designed to handle complex tabular datasets like the one in this competition, showcasing my ability to apply cutting-edge research to practical problems.

3.  **Sophisticated Ensemble Strategy:**
    The final prediction was not a simple average. I designed a carefully tuned, **two-layer weighted ensemble**:
    * **Layer 1:** The Neural Network and XGBoost predictions were first combined into a sub-ensemble.
    * **Layer 2:** The outputs from the sub-ensemble, the Ridge model, and the TabM model were then combined using a final set of optimized weights to produce the ultimate prediction. This hierarchical approach maximized predictive accuracy.

## Code Structure

This repository includes the following key components:

* `nn_train.ipynb`: Training script for the ResNet-like MLP model.
* `xgb_train.ipynb`: Training script for the XGBoost model.
* `ridge_train.ipynb`: Training script for the Ridge Regression model.
* `tabm_train.ipynb`: Training script for the advanced TabM deep learning model.
* `tanm_reference.py`: The core Python module defining the TabM model architecture.
* `ensemble.ipynb`: The final notebook that loads all trained models and executes the two-layer ensemble logic to generate predictions.

## Key Skills & Technologies

This project demonstrates my proficiency in:

-   **Advanced Machine Learning:** Sophisticated Ensemble Methods, Time-Series Forecasting, Gradient Boosting (XGBoost), Deep Learning (PyTorch).
-   **Model Implementation:** Ability to implement and adapt models from academic research papers (TabM).
-   **Quantitative Finance:** Practical understanding of market prediction tasks and evaluation metrics.
-   **Programming & Tools:** Python, Jupyter Notebooks.
-   **Data Science Libraries:** Pandas, Polars, NumPy, Scikit-learn.
