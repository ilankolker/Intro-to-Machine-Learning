# Gradient Descent & Regularized Logistic Regression 🚀

This project is part of a hands-on machine learning course at **Hebrew University**. In this exercise, you will implement a generic Gradient Descent algorithm and explore its performance on different objective functions, culminating in the minimization of a regularized logistic regression optimization problem.

---

## Project Overview

### 1. Gradient Descent Implementation 🔍
You will implement a generic Gradient Descent algorithm to minimize various objective functions:
- **Learning Rate Strategies**: Create a `FixedLR` class to manage a constant learning rate.
- **Objective Functions**: Implement `L1` and `L2` modules in the `modules.py` file to minimize using gradient descent.
- **Gradient Descent Algorithm**: Implement the `GradientDescent` class, allowing for different callbacks to investigate the algorithm's performance.

### 2. Investigating Fixed Learning Rates 📈
Explore the convergence of Gradient Descent over L1 and L2 objectives using fixed learning rates:
- **Descent Path Visualization**: Plot the descent paths for different learning rates and analyze the differences between L1 and L2 modules.
- **Convergence Rate Analysis**: Assess the convergence rate for each module and discuss the results.
- **Loss Evaluation**: Determine the lowest loss achieved when minimizing each of the modules.

### 3. Minimizing Regularized Logistic Regression 💼
Utilize your Gradient Descent implementation to solve a regularized logistic regression problem:
- **Logistic Module**: Implement the `LogisticModule` to calculate the negative log-likelihood and its derivative.
- **Regularized Module**: Create the `RegularizedModule` to incorporate both fidelity and regularization terms.
- **Logistic Regression Class**: Implement the `LogisticRegression` class, integrating the previously defined modules and allowing for easy extensions.

---

## Key Features 🌟
- **Gradient Descent Algorithm**: A flexible implementation to minimize various functions using different learning rates.
- **Objective Functions**: L1 and L2 regularization for robust model training.
- **Regularized Logistic Regression**: Implement logistic regression with regularization and evaluate using ROC curves.

---

## Packages Used 📦
- **Numpy**: For numerical computations and data handling.
- **Matplotlib**: For plotting graphs and visualizations.
- **Scikit-Learn**: For evaluating model performance and metrics.

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
    - For running the gradient descent investigation:
      ```bash
      python ex4/gradient_descent_investigation.py
      ```

---

## Key Concepts 💡
- **Gradient Descent**: An optimization algorithm for finding the minimum of a function by iteratively moving in the direction of steepest descent.
- **Regularization**: Techniques used to prevent overfitting by adding a penalty to the loss function.
- **Logistic Regression**: A statistical method for predicting binary classes, leveraging the logistic function.

---

## Visualizations 📊
- **Descent Paths**: Plot the paths taken by the Gradient Descent algorithm for different learning rates.
- **Convergence Rates**: Visualize the convergence rates for the L1 and L2 modules.
- **ROC Curves**: Generate ROC curves to evaluate the performance of the logistic regression model.

---

## Dataset 📊
- **South Africa Heart Disease Dataset**: Load and split the dataset to train and test sets for logistic regression evaluation.

---
