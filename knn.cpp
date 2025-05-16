#include <bits/stdc++.h>
using namespace std;

/*
K-Nearest Neighbors (knn.cpp)

Variables:
  n           is the number of training samples
  d           is the number of features per sample
  k           is the number of nearest neighbors to consult
  X           is an n×d array of training feature vectors
  y           is an array of length n of integer labels
  query       is a feature vector of length d to classify
  distances   is an array of n pairs (distance, label)

Algorithm steps:
  - Store all n training points with their labels.
  - For the query vector, compute its Euclidean distance to each training point.
  - Use partial sorting to find the k points with the smallest distances.
  - Count how many times each label appears among those k neighbors.
  - Return the label with the highest count.

Big-O analysis:
  - Distance computation    is O(n * d) via 
      for i in 0..n-1 for j in 0..d-1 dist += (X[i][j] - query[j])*(X[i][j] - query[j])
  - Selecting k smallest     is O(n) via nth_element(distances.begin(), distances.begin()+k, distances.end())
  - Voting among k           is O(k) via tally[ distances[i].second ]++
  Total time               is O(n*d + n + k)

Goal:
  Predict the class label of a new data point by majority vote among its k nearest neighbors in feature space.
*/


// Holds a feature vector and its integer label
struct DataPoint {
    vector<double> features;  // d-dimensional features
    int label;                // class label
};

// euclideanDist:
//   Compute Euclidean distance between two d-dimensional vectors.
//   - Time: O(d)
double euclideanDist(const vector<double>& a, const vector<double>& b) {
    double sumSq = 0.0;
    for (int i = 0; i < (int)a.size(); ++i) {
        double diff = a[i] - b[i];  // one subtraction
        sumSq += diff * diff;       // one multiply + one add
    }
    return sqrt(sumSq);             // one sqrt
}

// knnPredict:
//   Predicts label for 'query' using the k nearest neighbors in 'data'.
//   - Compute distances: O(n*d)
//   - Select k smallest:   O(n)
//   - Vote among k:        O(k)
//   - Total: O(n*d + n + k)
int knnPredict(const vector<DataPoint>& data,
               const vector<double>& query,
               int k)
{
    int n = data.size();
    vector<pair<double,int>> dists;
    dists.reserve(n);

    // 1) Distance computations (n * O(d))
    for (int i = 0; i < n; ++i) {
        double dist = euclideanDist(data[i].features, query);
        dists.emplace_back(dist, data[i].label);
    }

    // 2) Select k smallest distances (average O(n))
    nth_element(dists.begin(),
                dists.begin() + k,
                dists.end(),
                [](auto &A, auto &B){ return A.first < B.first; });

    // 3) Vote among the k labels (O(k))
    unordered_map<int,int> freq;
    for (int i = 0; i < k; ++i) {
        freq[dists[i].second]++;
    }

    // 4) Pick majority label (<= k comparisons)
    int bestLabel = freq.begin()->first;
    int bestCount = 0;
    for (auto& pr : freq) {
        if (pr.second > bestCount) {
            bestCount = pr.second;
            bestLabel = pr.first;
        }
    }
    return bestLabel;
}

int main() {
    int n, d, k;
    if (!(cin >> n >> d >> k)) {
        cerr << "Usage: n d k\n"
             << "Then n lines: d features and 1 label\n"
             << "Then 1 line: d features for query\n";
        return 1;
    }

    vector<DataPoint> dataset(n);
    for (int i = 0; i < n; ++i) {
        dataset[i].features.resize(d);
        for (int j = 0; j < d; ++j) {
            cin >> dataset[i].features[j];
        }
        cin >> dataset[i].label;
    }

    vector<double> query(d);
    for (int j = 0; j < d; ++j) {
        cin >> query[j];
    }

    int prediction = knnPredict(dataset, query, k);
    cout << "Predicted label: " << prediction << "\n";
    return 0;
}

// // “Training” step: just store the dataset
// function trainKNN(X[n][d], y[n]) -> (storedX, storedY):
//   return (X, y)
//
// // Prediction step: find the majority label among the k nearest neighbors
// function predictKNN(storedX[n][d], storedY[n], k, query[d]) -> predictedLabel:
//   // 1) Compute distances to all training points
//   list = empty list of (distance, label)
//   for i in 0..n-1:
//     dist = EuclideanDistance(storedX[i], query)   // sqrt( sum_j (X[i][j]–query[j])^2 )
//     append (dist, storedY[i]) to list
//
//   // 2) Partially sort so the first k entries are the smallest distances
//   nth_element(list.begin(), list.begin()+k, list.end(), compare by .distance )
//
//   // 3) Vote among the k nearest labels
//   voteCount = empty map<label, count>
//   for i in 0..k-1:
//     voteCount[ list[i].label ] += 1
//
//   // 4) Return the label with the highest vote
//   return argmax_label voteCount[label]
