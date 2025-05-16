#include <bits/stdc++.h>
using namespace std;

/*
Gaussian Naive Bayes (gnb.cpp)

Variables:
  n            is the number of training samples
  d            is the number of features per sample
  classes      is the list of unique class labels of size c
  count        is an array of length c storing sample counts per class
  priors       is an array of length c storing class prior probabilities
  mean         is a c×d matrix of per-class feature means
  var          is a c×d matrix of per-class feature variances
  data         holds n samples, each with d features and an integer label
  query        is a feature vector of length d to classify

Algorithm steps:
  - Identify unique classes by scanning every sample’s label.
  - Accumulate feature sums per class by looping over all samples and features.
  - Divide sums by counts to compute each class’s prior and feature means.
  - Accumulate squared deviations from the mean per class by looping again over samples and features.
  - Divide those sums of squares by counts to finalize variances.
  - For the query, compute for each class the log of its prior plus the sum of log Gaussian probabilities across all features.
  - Return the class with the highest total log-score.

Big-O analysis:
  - Identifying classes          is O(n)     via sc.insert(pt.label)
  - Summing feature values       is O(n * d) via mean[i][j] += data[i].features[j]
  - Computing priors and means   is O(c * d) via priors[i] = count[i]/n and mean[i][j] /= count[i]
  - Summing squared deviations   is O(n * d) via var[i][j] += diff * diff
  - Finalizing variances         is O(c * d) via var[i][j] /= count[i]
  - Inference per query          is O(c * d) via gaussianLogProb(query[j], mean[i][j], var[i][j])
  Total training complexity      is O(n * d + c * d)
  Total inference complexity     is O(c * d)

Goal:
  - Learn Gaussian distributions (mean and variance) for each feature conditioned on class, then predict the most likely class label for a new feature vector by choosing the highest posterior probability.
*/


// Holds a feature vector and its integer label
struct DataPoint {
    vector<double> features;  // d-dimensional features
    int label;                // class label
};

// ----------------------------------------------------------------------------
// trainNaiveBayes:
//   Learns priors, per-class means & variances from the data.
//   - Time: O(n*d + c*d + n + c)
// ----------------------------------------------------------------------------
void trainNaiveBayes(const vector<DataPoint>& data,
                     vector<int>& classes,
                     vector<double>& priors,
                     vector<vector<double>>& mean,
                     vector<vector<double>>& var)
{
    int n = data.size();
    int d = data[0].features.size();

    // 1) Identify unique classes      (<= n ops)
    unordered_set<int> sc;
    for (const auto& pt : data) sc.insert(pt.label);
    classes.assign(sc.begin(), sc.end());
    int c = classes.size();

    // Map class label -> index
    unordered_map<int,int> idx;
    for (int i = 0; i < c; ++i)
        idx[classes[i]] = i;

    // 2) Initialize accumulators
    vector<int> count(c, 0);
    priors.assign(c, 0.0);
    mean.assign(c, vector<double>(d, 0.0));
    var.assign(c,  vector<double>(d, 0.0));

    // 3) Sum features per class      (n*d ops)
    for (const auto& pt : data) {
        int i = idx[pt.label];
        count[i]++;
        for (int j = 0; j < d; ++j)
            mean[i][j] += pt.features[j];
    }

    // 4) Finalize priors & means     (c + c*d ops)
    for (int i = 0; i < c; ++i) {
        priors[i] = double(count[i]) / n;
        for (int j = 0; j < d; ++j)
            mean[i][j] /= count[i];
    }

    // 5) Sum squared diffs           (n*d ops)
    for (const auto& pt : data) {
        int i = idx[pt.label];
        for (int j = 0; j < d; ++j) {
            double diff = pt.features[j] - mean[i][j];
            var[i][j] += diff * diff;
        }
    }

    // 6) Finalize variances          (c*d ops)
    for (int i = 0; i < c; ++i)
        for (int j = 0; j < d; ++j)
            var[i][j] /= count[i];
}

// ----------------------------------------------------------------------------
// gaussianLogProb:
//   Computes log of Gaussian PDF for value x.
//   - O(1) per feature
// ----------------------------------------------------------------------------
double gaussianLogProb(double x, double mu, double sigma2) {
    static const double PI = 3.141592653589793;
    double diff = x - mu;
    return -0.5 * log(2 * PI * sigma2)
           - (diff * diff) / (2 * sigma2);
}

// ----------------------------------------------------------------------------
// naiveBayesPredict:
//   Predicts class label for 'query'.
//   - Time: O(c*d)
// ----------------------------------------------------------------------------
int naiveBayesPredict(const vector<double>& query,
                      const vector<int>& classes,
                      const vector<double>& priors,
                      const vector<vector<double>>& mean,
                      const vector<vector<double>>& var)
{
    int c = classes.size();
    int d = query.size();
    double bestLP = -1e308;
    int bestClass = classes[0];

    // Compute log-posterior for each class (c*d ops)
    for (int i = 0; i < c; ++i) {
        double lp = log(priors[i]);  // prior term
        for (int j = 0; j < d; ++j) {
            lp += gaussianLogProb(query[j], mean[i][j], var[i][j]);
        }
        if (lp > bestLP) {
            bestLP = lp;
            bestClass = classes[i];
        }
    }
    return bestClass;
}

// ----------------------------------------------------------------------------
// main:
//   Reads input, trains the classifier, predicts one query, prints result.
// ----------------------------------------------------------------------------
int main() {
    int n, d;
    if (!(cin >> n >> d)) {
        cerr << "Usage: <n> <d>\\n"
             << "Then n lines of d features and 1 label, then 1 query line of d features.\n";
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

    vector<int> classes;
    vector<double> priors;
    vector<vector<double>> mean, var;

    trainNaiveBayes(dataset, classes, priors, mean, var);

    int prediction = naiveBayesPredict(query, classes, priors, mean, var);
    cout << "Predicted label: " << prediction << "\n";

    return 0;
}

// // Train the model: compute priors, means, and variances
// function trainNaiveBayes(X[n][d], y[n]) -> (priors[C], means[C][d], vars[C][d]):
//   for each class c in unique(y):
//     prior[c] = count(y==c) / n
//     for j in 0..d-1:
//       values = [ X[i][j] for i where y[i]==c ]
//       mean[c][j] = average(values)
//       var[c][j]  = variance(values)
//   return priors, means, vars
//
// // Predict the class of one new example
// function predictGNB(query[d], priors, means, vars) -> predictedClass:
//   for each class c:
//     score[c] = log(prior[c])
//     for j in 0..d-1:
//       score[c] += log GaussianPDF(query[j], mean[c][j], var[c][j])
//   return argmax_c score[c]
