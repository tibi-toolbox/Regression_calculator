#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace std;

ofstream g("result.txt");
ifstream f("input.txt");

// Function to calculate the Logarithmic regression coefficients
vector<double> logarithmicRegression(const vector<double>& x, const vector<double>& y)
{
    int n = x.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum_x += log(x[i]);
        sum_y += y[i];
        sum_xy += log(x[i]) * y[i];
        sum_x2 += log(x[i]) * log(x[i]);
    }

    double b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    double a = (sum_y - b * sum_x) / n;

    return {a, b};
}

// Function to calculate the Exponential regression coefficients
vector<double> exponentialRegression(const vector<double>& x, const vector<double>& y)
{
    int n = x.size();
    double sum_x = 0.0, sum_y = 0.0, sum_x2y = 0.0, sum_x2 = 0.0;
    for (int i = 0; i < n; i++) sum_x += x[i],sum_y += log(y[i]),sum_x2y += x[i] * log(y[i]),sum_x2 += x[i] * x[i];

    double b = (n * sum_x2y - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    double a = (sum_y - b * sum_x) / n;

    return { exp(a), b };
}

// Function to calculate the Power regression coefficients
vector<double> powerRegression(const vector<double>& x, const vector<double>& y)
{
    int n = x.size();
    double sum_logx = 0.0, sum_logy = 0.0, sum_logxlogy = 0.0, sum_logx2 = 0.0;
    for (int i = 0; i < n; i++) sum_logx += log(x[i]),sum_logy += log(y[i]),sum_logxlogy += log(x[i]) * log(y[i]),sum_logx2 += log(x[i]) * log(x[i]);

    double b = (n * sum_logxlogy - sum_logx * sum_logy) / (n * sum_logx2 - sum_logx * sum_logx);
    double a = (sum_logy - b * sum_logx) / n;

    return { exp(a), b };
}

// Function to calculate the Polynomial regression coefficients
vector<double> polynomialRegression(int degree, const vector<double>& x, const vector<double>& y)
{
    int n = x.size();
    vector<double> coefficients(degree + 1, 0.0);

    vector<vector<double>> X(n, vector<double>(degree + 1, 1.0));

    #pragma omp parrallel for
    for (int i = 0; i < n; i++) for (int j = 1; j <= degree; j++) X[i][j] = x[i] * X[i][j - 1];

    vector<vector<double>> XT(degree + 1, vector<double>(n, 0.0));
    #pragma omp parrallel for
    for (int i = 0; i < n; i++) for (int j = 0; j <= degree; j++) XT[j][i] = X[i][j];

    vector<vector<double>> XTX(degree + 1, vector<double>(degree + 1, 0.0));
    for (int i = 0; i <= degree; i++)
    {
        for (int j = 0; j <= degree; j++)
        {
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (int k = 0; k < n; k++) sum += XT[i][k] * X[k][j];
            XTX[i][j] = sum;
        }
    }

    vector<double> XTY(degree + 1, 0.0);
    for (int i = 0; i <= degree; i++)
    {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < n; j++) sum += XT[i][j] * y[j];
        XTY[i] = sum;
    }

    for (int i = 0; i <= degree; i++)
    {
        for (int j = i + 1; j <= degree; j++)
        {
            double ratio = XTX[j][i] / XTX[i][i];
            #pragma omp parrallel for
            for (int k = 0; k <= degree; k++) XTX[j][k] -= ratio * XTX[i][k];
            XTY[j] -= ratio * XTY[i];
        }
    }

    coefficients[degree] = XTY[degree] / XTX[degree][degree];

    for (int i = degree - 1; i >= 0; i--)
    {
        coefficients[i] = XTY[i];
        for (int j = i + 1; j <= degree; j++) coefficients[i] -= XTX[i][j] * coefficients[j];
        coefficients[i] /= XTX[i][i];
    }

    return coefficients;
}


// Function to calculate the coefficient of determination (R-squared)
double coefficientOfDetermination(const vector<double>& x, const vector<double>& y, const vector<double>& coefficients, const string& regressionType)
{
    int n = x.size();
    double y_mean = 0.0;
    for (int i = 0; i < n; i++) y_mean += y[i];
    y_mean /= n;

    double y_hat_mean = 0.0;
    double ssr = 0.0, sst = 0.0;

    if (regressionType == "Polynomial")
    {
        for (int i = 0; i < n; i++)
        {
            double y_hat = 0.0;
            for (int j = 0; j < coefficients.size(); j++)
            {
                y_hat += coefficients[j] * pow(x[i], j);
            }
            ssr += pow(y_hat - y[i], 2);
            sst += pow(y[i] - y_mean, 2);
            y_hat_mean += y_hat;
        }
    }
    else if (regressionType == "Exponential")
    {
        for (int i = 0; i < n; i++)
        {
            double y_hat = coefficients[0] * exp(coefficients[1] * x[i]);
            ssr += pow(y_hat - y[i], 2);
            sst += pow(y[i] - y_mean, 2);
            y_hat_mean += y_hat;
        }
    }
    else if (regressionType == "Logarithmic")
    {
        for (int i = 0; i < n; i++)
        {
            if (x[i] <= 0) continue;
            double y_hat = coefficients[0] + coefficients[1] * log(x[i]);
            ssr += pow(y_hat - y[i], 2);
            sst += pow(y[i] - y_mean, 2);
            y_hat_mean += y_hat;
        }
    }
    else if (regressionType == "Power")
    {
        for (int i = 0; i < n; i++)
        {
            if (x[i] <= 0 || coefficients[0] < 0 || coefficients[1] < 0) continue; // Skip cases where x is <= 0 or coefficients are negative
            double y_hat = coefficients[0] * pow(x[i], coefficients[1]);
            ssr += pow(y_hat - y[i], 2);
            sst += pow(y[i] - y_mean, 2);
            y_hat_mean += y_hat;
        }
    }

    y_hat_mean /= n;

    double r2 = 1.0 - (ssr / sst);
    r2 = max(0.0, min(r2, 1.0)); // Clamp R-squared to be between 0 and 1
    return r2;
}


// Function to print the regression equation
void printRegressionEquation(const vector<double>& coefficients, const string& type)
{
    int degree = coefficients.size() - 1;
    cout << "Best Regression: " << type << " Regression" << "\n";
    g << "Best Regression: " << type << " Regression" << "\n";
    cout << "Equation: f(x) = ";
    g << "Equation: f(x) = ";
    if (type == "Polynomial")
    {
        for (int i = degree; i >= 0; i--)
        {
            if (!i) cout << coefficients[i] << "\n",g << coefficients[i] << "\n";
            else if (i == 1) cout << coefficients[i] << "x + ",g << coefficients[i] << "x + ";
            else cout << coefficients[i] << "x^" << i << " + ",g << coefficients[i] << "x^" << i << " + ";
        }
    }
    else if (type == "Exponential") cout << coefficients[0] << " * e^ ( " << coefficients[1] << " * x)" << "\n", g<< coefficients[0] << " * e^ ( " << coefficients[1] << " * x)" << "\n";

    else if (type == "Logarithmic") cout << coefficients[1] << " * log(x) + " << coefficients[0] << "\n", g<< coefficients[1] << " * log(x) + " << coefficients[0] << "\n";

    else if (type == "Power") cout << coefficients[0] << " * x^" << coefficients[1] << "\n", g<< coefficients[0] << " * x^" << coefficients[1] << "\n";

}

int main()
{
    int n,cnt=0;

    string filename = "input.txt"; // Replace with your file name

    if (f.is_open())
    {
        f >> n;
        cnt=1;
    }

    else cout << "Enter the number of data points: ",cin >> n;

    vector<double> x(n);
    vector<double> y(n);

    if (cnt==1)
    {
        for (int i = 0; i < n; i++) f >> x[i] >> y[i];
    }

    else
    {
        cout << "Enter the data points in the format (x, y):" << "\n";
        for (int i = 0; i < n; i++) cin >> x[i] >> y[i];
    }

    cnt=0;

    // Perform different regressions and determine the best one
    vector<pair<string, double>> regressions;

    // Logarithmic regression
    vector<double> logCoefficients = logarithmicRegression(x, y);
    double logR2 = coefficientOfDetermination(x, y, logCoefficients, "Logarithmic");
    regressions.push_back(make_pair("Logarithmic", logR2));

    // Exponential regression
    vector<double> expCoefficients = exponentialRegression(x, y);
    double expR2 = coefficientOfDetermination(x, y, expCoefficients, "Exponential");
    regressions.push_back(make_pair("Exponential", expR2));

    // Power regression
    vector<double> powerCoefficients = powerRegression(x, y);
    double powerR2 = coefficientOfDetermination(x, y, powerCoefficients, "Power");
    regressions.push_back(make_pair("Power", powerR2));

    double bestPolyR2 = -1.0;
    vector<double> bestPolyCoefficients;
    int bestDegree = -1;

    // Polynomial regression with degree from 1 to 101
    for (int degree = 1; degree <= 2^64 && bestPolyR2!=1; degree++)
    {
        vector<double> polyCoefficients = polynomialRegression(degree, x, y);
        double polyR2 = coefficientOfDetermination(x, y, polyCoefficients,"Polynomial");
        if(isnan( polyCoefficients[0])) break;

        if (polyR2 > bestPolyR2)
        {
            bestPolyR2 = polyR2;
            bestPolyCoefficients = polyCoefficients;
            bestDegree = degree;
        }
    }
    regressions.push_back(make_pair("Polynomial", bestPolyR2));

    // Sort the regressions based on R^2 in ascending order
    sort(regressions.begin(), regressions.end(), [](const pair<string, double>& a, const pair<string, double>& b)
    {
        return a.second < b.second;
    });

    // Print the regression equations in order from worst to most accurate
    cout << "\n\nRegression Equations in order of accuracy (worst to best):\n\n" << "\n";
    g << "\Regression Equations in order of accuracy (worst to best):\n\n" << "\n";
    for (const auto& regression : regressions)
    {
        if (regression.second > 0.0)
        {
            cout << regression.first << " Regression: ";
            g << regression.first << " Regression: ";
            if (regression.first == "Logarithmic") printRegressionEquation(logCoefficients, regression.first);
            else if (regression.first == "Exponential") printRegressionEquation(expCoefficients, regression.first);
            else if (regression.first == "Power") printRegressionEquation(powerCoefficients, regression.first);
            else if (regression.first == "Polynomial") printRegressionEquation(bestPolyCoefficients, regression.first);

            cout << "Accuracy: " << (regression.second * 100.0) << "%" << "\n\n";
            g << "Accuracy: " << (regression.second * 100.0) << "%" << "\n\n";
            cnt++;
        }
    }
    if (!cnt) cout<<"None of the regressions fit the points in any way.\n\n";

    return 0;
}
