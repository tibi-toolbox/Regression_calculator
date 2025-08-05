# Regression_calculator
A C++ program that performs Logarithmic, Exponential, Power, and Polynomial regression on a dataset and identifies the best-fitting model using R² (coefficient of determination).


#This C++ program performs curve fitting using four types of regression models:

o Logarithmic

o Exponential

o Power

o Polynomial (with dynamic degree search, up to 100)



It analyzes a dataset of (x, y) values and identifies the best-fitting model by calculating the coefficient of determination (R²) for each. The results are printed to both the console and an output file.





  
#  ⚙ How It Works

  Input Handling:

The program reads data from input.txt (if available).
If not found, it prompts the user to input data manually via the console.

#  Regression Models:

o Logarithmic Regression: Fits the model y = a + b * log(x)

o Exponential Regression: Fits y = a * e^(b * x)

o Power Regression: Fits y = a * x^b

o Polynomial Regression: Fits y = a₀ + a₁x + a₂x² + ... + aₙxⁿ (with n as high as 100)

o Polynomial degree increases dynamically until the best fit is found or the coefficients become invalid.


#  Model Evaluation:

Each regression is evaluated using the coefficient of determination (R²), which measures how well the model fits the data.
R² is clamped between 0 and 1 to avoid invalid values.

#  Best Fit Selection:
All models are ranked by their R² score.

#  Output:

Console output with formatted regression equations and accuracies.
Results are also written to a file named result.txt.
