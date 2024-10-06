# Classifier Comparison & Evaluation 🔍

This project is part of a hands-on machine learning course at **Hebrew University**. In this exercise, you will implement various classifiers and compare their performances across different datasets. The focus is on understanding the behavior of classifiers in both linearly separable and inseparable scenarios.

---

## Project Overview

### 1. Perceptron Classifier 🧠
You will implement the Perceptron algorithm to classify linearly separable data:
- **Implementation**: Create the `misclassification_error` function to evaluate the classifier's performance.
- **Training**: Fit the Perceptron model on the `linearly_separable.npy` dataset.
- **Visualization**: Plot the decision boundary and analyze the results.

### 2. Evaluation of Classifiers 📊
Next, you will assess the Perceptron's performance on a more challenging dataset:
- **Inseparable Data**: Run the Perceptron on `linearly_inseparable.npy` and compare the loss over iterations.
- **Discussion**: Explain the differences in behavior between the two datasets in terms of objectives and parameter space.

### 3. Bayes Classifiers 🌌
You will implement and evaluate Bayes classifiers, including:
- **Gaussian Naive Bayes**: Fit the Gaussian Naive Bayes model to data.
- **LDA Classifier**: Implement the Linear Discriminant Analysis classifier.
- **Comparison**: Analyze the classifiers using the `gaussians1.npy` and `gaussians2.npy` datasets, focusing on how well they fit the data and what the plots reveal about the underlying distributions.

---

## Key Features 🌟
- **Classifier Implementation**: Complete implementations of the Perceptron, Gaussian Naive Bayes, and LDA classifiers.
- **Performance Evaluation**: Tools to assess and visualize classifier performance through loss plots.
- **Comparative Analysis**: Explore how different classifiers handle various data distributions.

---

## Datasets 📊
- **Linearly Separable Data**: `linearly_separable.npy` - Dataset for testing the Perceptron on easy cases.
- **Linearly Inseparable Data**: `linearly_inseparable.npy` - Dataset for testing the Perceptron on difficult cases.
- **Gaussian Data 1**: `gaussians1.npy` - Data for evaluating Gaussian Naive Bayes and LDA.
- **Gaussian Data 2**: `gaussians2.npy` - Additional dataset for comparative analysis.

---

## Instructions 🛠️

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ilankolker/Intro-to-Machine-Learning.git
    ```

2. **Install dependencies** (if needed):
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the code**:
    - For Perceptron implementation:
      ```bash
      python ex2/perceptron.py
      ```
    - For Bayes classifiers:
      ```bash
      python ex2/bayes_classifiers.py
      ```

---

## Key Concepts 💡
- **Perceptron Algorithm**: A linear classifier that updates weights based on misclassification.
- **Bayes Classifiers**: Probabilistic classifiers based on Bayes' theorem.
- **Loss Functions**: Metrics to evaluate classifier performance during training and testing.
- **Data Visualization**: Tools to visualize the decision boundaries and classifier performance.

---

## Visualizations 📊
- **Decision Boundary**: Visualize how the Perceptron separates linearly separable data.
- **Loss Progression**: Track the Perceptron’s loss over iterations for both datasets.
- **Classifier Comparison**: Scatter plots to visualize classifier predictions vs. true classes for Gaussian data.

---
