# Regression_Calculator

**Regression_Calculator** is a C++ program that fits four regression models to your dataset and selects the best one using the coefficient of determination (R²).

---

## Description

This tool performs curve fitting on (x, y) data points using:
- **Logarithmic Regression** (`y = a + b·log(x)`)
- **Exponential Regression** (`y = a·e^(b·x)`)
- **Power Regression** (`y = a·x^b`)
- **Polynomial Regression** (`y = a₀ + a₁x + a₂x² + … + aₙxⁿ`), with degree up to 100.

Each model’s quality is measured by **R²** (clamped to [0, 1]). Results are printed on the console and saved to `result.txt`.

---

## ⚙ How It Works

1. **Input**  
   - Reads `(x, y)` pairs from `input.txt` if available.  
   - Otherwise, prompts for manual entry.

2. **Regression**  
   - Fits each of the four models to the data.  
   - Dynamically raises polynomial degree until the best fit or limit.

3. **Evaluation**  
   - Computes R² for every model.  
   - Clamps R² between 0 and 1.

4. **Selection**  
   - Ranks models by R² and picks the highest.

5. **Output**  
   - Displays all equations with R² in the console.  
   - Writes the best fit and its R² to `result.txt`.

---

##  Usage

```bash
g++ regression_calculator.cpp -o regression_calculator -std=c++11 -O2
./regression_calculator
