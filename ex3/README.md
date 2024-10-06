# Decision Stump & AdaBoost Implementation ⚡

This project is part of a hands-on machine learning course at **Hebrew University**. In this exercise, you will implement a decision stump (a decision tree of depth 1) and the AdaBoost algorithm to enhance model performance through boosting techniques.

---

## Project Overview

### 1. Boosting with AdaBoost 🌟
You will implement the AdaBoost algorithm and a decision stump as its weak learner:
- **Decision Stump**: Create a `DecisionStump` class that selects the best feature and threshold for classification.
- **AdaBoost Algorithm**: Implement the `AdaBoost` class, which will aggregate multiple weak learners to improve classification accuracy.

### 2. Experimental Setup 🧪
You will conduct a series of experiments to evaluate the performance of your AdaBoost implementation:
- **Error Analysis**: Train an AdaBoost ensemble on clean data (no noise) and plot the training and test errors as a function of the number of fitted learners.
- **Decision Boundaries**: Visualize the decision boundaries of the ensemble at various iterations and analyze the results.
- **Ensemble Evaluation**: Identify the ensemble size with the lowest test error and visualize the decision surface along with the test set.
- **Weight Visualization**: Analyze the weights from the last iteration to understand which samples are easier or more challenging for the classifier.
- **Noise Impact**: Repeat the experiments with added noise and discuss the results in the context of the bias-variance tradeoff.

---

## Key Features 🌟
- **Decision Stump Implementation**: Create a decision tree with depth 1 for classification tasks.
- **AdaBoost Implementation**: Aggregate weak learners to form a robust classifier.
- **Visualization Tools**: Utilize Plotly and Matplotlib for visualizing errors and decision boundaries.

---

## Packages Used 📦
- **Numpy**: For numerical computations and data handling.
- **Matplotlib**: For plotting graphs and visualizations.
- **Plotly**: For interactive visualizations.
- **itertools**: For efficient looping in experiments.

---

## Datasets 📊
- **Generated Data**: Use the provided `generate_data` function to create training and test samples, with varying levels of noise.

---

## Instructions 🛠️

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ilankolker/Intro-to-Machine-Learning.git
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the code**:
    - For running the AdaBoost scenario:
      ```bash
      python exercise3/adaboost_scenario.py
      ```

---

## Key Concepts 💡
- **Boosting**: A machine learning ensemble technique that combines weak learners to create a strong classifier.
- **Decision Stump**: A simple classifier that makes decisions based on a single feature.
- **Model Evaluation**: Assessing model performance using training and test errors, and analyzing decision boundaries.

---

## Visualizations 📊
- **Error Plots**: Visualize training and test errors as a function of the number of fitted learners.
- **Decision Boundaries**: Examine how the decision surface evolves as more weak learners are added.
- **Sample Weights**: Analyze the distribution of sample weights to identify challenging and easy samples for classification.
- **Noise Analysis**: Evaluate how noise affects model performance and decision boundaries, focusing on bias-variance tradeoff.

---
