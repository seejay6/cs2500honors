#include <bits/stdc++.h>
using namespace std;

/*
Perceptron Classifier (prc.cpp)

Variables:
  n   is the number of training samples
  d   is the number of features per sample
  E   is the number of epochs to train
  lr  is the learning rate
  X   is an n×d array of training feature vectors
  y   is an array of length n of labels (+1 or –1)
  w   is the weight vector of length d
  b   is the bias term (scalar)

Algorithm steps (repeated for E epochs):
  - For each training sample i:
    compute the activation as the dot product of w and X[i] plus b
    predict +1 if activation => 0, otherwise predict –1
    if the prediction does not match y[i], adjust w and b to correct the error

Big-O analysis:
  - Training is O(E * n * d)
    via nested loops over epochs, samples, and features:
      activation += w[j] * X[i][j];
      w[j] += lr * y[i] * X[i][j];
      b   += lr * y[i];
  - Inference is O(d)
    via one loop over features:
      activation += w[j] * query[j];

Goal:
  Learn a linear decision boundary that classifies inputs as +1 or –1 by iteratively correcting mistakes.
*/


// Holds one training example
struct DataPoint {
    vector<double> features;
    int label;
};

// trainPerceptron:
//   Learns weight vector w and bias b over E epochs using learning rate lr.
//   - Time: O(E*n*d)
void trainPerceptron(const vector<DataPoint>& data,
                     vector<double>& w,
                     double& b,
                     int E,
                     double lr)
{
    int n = data.size();
    int d = data[0].features.size();
    // initialize
    w.assign(d, 0.0);
    b = 0.0;

    // epochs
    for (int epoch = 0; epoch < E; ++epoch) {
        for (const auto& pt : data) {
            // compute activation a = w*x + b
            double a = b;
            for (int j = 0; j < d; ++j)
                a += w[j] * pt.features[j];

            // predict
            int y_pred = (a >= 0 ? 1 : -1);

            // update if wrong
            if (y_pred != pt.label) {
                for (int j = 0; j < d; ++j)
                    w[j] += lr * pt.label * pt.features[j];
                b += lr * pt.label;
            }
        }
    }
}

// perceptronPredict:
//   Returns +1 or -1 for input x given trained w and b.
//   - Time: O(d)
int perceptronPredict(const vector<double>& x,
                      const vector<double>& w,
                      double b)
{
    double a = b;
    for (int j = 0; j < (int)x.size(); ++j)
        a += w[j] * x[j];
    return (a >= 0 ? 1 : -1);
}

int main() {
    int n, d, E;
    double lr;
    if (!(cin >> n >> d >> E >> lr)) {
        cerr << "Usage: n d E lr\n"
             << "Then " << n << " lines: d features and +1/-1 label\n"
             << "Then 1 line: d features for query\n";
        return 1;
    }

    vector<DataPoint> dataset(n);
    for (int i = 0; i < n; ++i) {
        dataset[i].features.resize(d);
        for (int j = 0; j < d; ++j)
            cin >> dataset[i].features[j];
        cin >> dataset[i].label;
    }

    vector<double> query(d);
    for (int j = 0; j < d; ++j)
        cin >> query[j];

    vector<double> weights;
    double bias;
    trainPerceptron(dataset, weights, bias, E, lr);

    int prediction = perceptronPredict(query, weights, bias);
    cout << "Predicted label: " << prediction << "\n";
    return 0;
}

// // Train a binary linear classifier
// function trainPerceptron(X[n][d], y[n], epochs, learningRate) -> (w[d], bias):
//   // 1) Initialize weights and bias
//   w = zero vector of length d
//   bias = 0
//   // 2) Repeat for each epoch
//   for epoch in 1..epochs:
//     // 3) Loop over each training sample
//     for i in 0..n-1:
//       // 4) Compute activation
//       activation = dot(w, X[i]) + bias
//       // 5) Predict class (+1 or –1)
//       prediction = (activation >= 0 ? +1 : -1)
//       // 6) Update if wrong
//       if prediction != y[i]:
//         for j in 0..d-1:
//           w[j] += learningRate * y[i] * X[i][j]
//         bias += learningRate * y[i]
//   return (w, bias)
//
// // Predict the label for a new example
// function predictPerceptron(w[d], bias, query[d]) -> predictedLabel:
//   activation = dot(w, query) + bias
//   if activation >= 0:
//     return +1
//   else:
//     return -1
