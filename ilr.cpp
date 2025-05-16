#include <bits/stdc++.h>
using namespace std;

/*
Linear Regression (ilr.cpp)

Variables:
  n            is the number of training samples
  d            is the number of features per sample
  A            is the d×d matrix that will hold X^T X
  b            is the length-d vector that will hold X^T y
  w            is the length-d weight vector solving A w = b
  query        is the new feature vector of length d to predict

Algorithm steps:
  - Assemble the normal equations by accumulating X^T X into A and X^T y into b.
  - Solve the d×d linear system A w = b using Gauss-Jordan elimination.
  - Compute the predicted value as the dot product of w and the query vector.

Big-O analysis:
  - Building A (X^T X)     is O(n * d^2) via A[j][k] += X[i][j] * X[i][k]
  - Building b (X^T y)     is O(n * d)   via b[j]     += X[i][j] * y[i]
  - Solving the system     is O(d^3)     via triple-nested pivot and elimination loops
  - Prediction             is O(d)       via sum += w[j] * query[j]
  Total training time      is O(n * d^2 + d^3)
  Total inference time     is O(d)

Goal:
  - Learn weights w by minimizing squared error in closed form, then predict a continuous target value for a new feature vector.
*/


// Solve A*w = b by Gauss–Jordan elimination (in-place on copies)
// Returns solution vector w of size d
vector<double> solveLinearSystem(vector<vector<double>> A,
                                 vector<double> b) {
    int d = A.size();
    const double EPS = 1e-12;
    for (int i = 0; i < d; ++i) {
        // Pivot selection
        int maxRow = i;
        double maxVal = fabs(A[i][i]);
        for (int r = i+1; r < d; ++r) {
            double v = fabs(A[r][i]);
            if (v > maxVal) {
                maxVal = v;
                maxRow = r;
            }
        }
        if (maxVal < EPS) continue;  // singular or near-zero pivot; skip
        swap(A[i], A[maxRow]);
        swap(b[i], b[maxRow]);
        // Normalize pivot row
        double diag = A[i][i];
        for (int col = i; col < d; ++col) A[i][col] /= diag;
        b[i] /= diag;
        // Eliminate in all other rows
        for (int r = 0; r < d; ++r) {
            if (r == i) continue;
            double factor = A[r][i];
            for (int col = i; col < d; ++col)
                A[r][col] -= factor * A[i][col];
            b[r] -= factor * b[i];
        }
    }
    return b;
}

int main() {
    int n, d;
    if (!(cin >> n >> d)) {
        cerr << "Usage: n d\n"
             << "Then n lines: d features and 1 target value\n"
             << "Then 1 line: d features for query\n";
        return 1;
    }

    // Read training data
    vector<vector<double>> X(n, vector<double>(d));
    vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j)
            cin >> X[i][j];
        cin >> y[i];
    }

    // Read query point
    vector<double> query(d);
    for (int j = 0; j < d; ++j)
        cin >> query[j];

    // Build normal equations: XtX and Xty
    vector<vector<double>> XtX(d, vector<double>(d, 0.0));
    vector<double> Xty(d, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            Xty[j] += X[i][j] * y[i];
            for (int k = 0; k < d; ++k)
                XtX[j][k] += X[i][j] * X[i][k];
        }
    }

    // Solve for weights w
    vector<double> w = solveLinearSystem(XtX, Xty);

    // Predict on query
    double ypred = 0.0;
    for (int j = 0; j < d; ++j)
        ypred += w[j] * query[j];

    cout << "Predicted value: " << ypred << "\n";
    return 0;
}

// // Train by solving the normal equations
// function trainLinearRegression(X[n][d], y[n]) -> w[d]:
//   A = zero_matrix(d, d)
//   b = zero_vector(d)
//   for i in 0..n-1:
//     for j in 0..d-1:
//       b[j] += X[i][j] * y[i]
//       for k in 0..d-1:
//         A[j][k] += X[i][j] * X[i][k]
//   w = solveLinearSystem(A, b)  // e.g. Gauss–Jordan elimination
//   return w
//
// // Predict a new example
// function predictLinearRegression(w[d], query[d]) -> yhat:
//   yhat = 0
//   for j in 0..d-1:
//     yhat += w[j] * query[j]
//   return yhat
