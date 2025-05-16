#include <bits/stdc++.h>
using namespace std;

/*
K-Means Clustering (kms.cpp)

Variables:
  n         is the number of data points
  d         is the dimension of each point
  k         is the number of clusters
  I         is the number of iterations
  points    is an n×d array of input points
  centroids is a k×d array of current cluster centers
  labels    is an array of length n storing each point’s cluster index

Algorithm steps (repeated I times):
  - Assignment step
    Each point is compared to every centroid and assigned to the nearest one.
  - Update step
    Each centroid is moved to the average of all points assigned to it.

Big-O analysis:
  - Assignment is O(n * k * d) 
    (nested loops: for i in 0..n-1, for c in 0..k-1, for j in 0..d-1 compute 
      dist += (points[i][j] - centroids[c][j]) * (points[i][j] - centroids[c][j]);)
  - Update is O(n * d) 
    (accumulate sums via sums[labels[i]][j] += points[i][j] for i in 0..n-1 and j in 0..d-1,
     then recompute centroids via centroids[c][j] = sums[c][j] / count[c] for c in 0..k-1 and j in 0..d-1)
  Total per iteration: O(n * k * d)
  Total over I iterations: O(I * n * k * d)

Goal:
  Partition the input points into k clusters by iteratively assigning points to the nearest centroid and updating centroids to the mean of their assigned points, outputting the final centroids and labels.
*/


int main() {
    int n, d, k, I;
    if (!(cin >> n >> d >> k >> I)) {
        cerr << "Usage: n d k I\n"
             << "Then n lines: d features each\n";
        return 1;
    }

    // Read data points
    vector<vector<double>> points(n, vector<double>(d));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            cin >> points[i][j];
        }
    }

    // Initialize centroids by randomly selecting k distinct points
    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);
    mt19937 gen(static_cast<unsigned>(std::time(nullptr)));
    shuffle(indices.begin(), indices.end(), gen);

    vector<vector<double>> centroids(k, vector<double>(d));
    for (int c = 0; c < k; ++c) {
        centroids[c] = points[indices[c]];
    }

    vector<int> labels(n);
    vector<vector<double>> newCentroids(k, vector<double>(d));
    vector<int> counts(k);

    // K-Means iterations
    for (int iter = 0; iter < I; ++iter) {
        // 1) Assignment step
        for (int i = 0; i < n; ++i) {
            int bestCluster = 0;
            double bestDist = numeric_limits<double>::infinity();
            for (int c = 0; c < k; ++c) {
                double dist = 0.0;
                for (int j = 0; j < d; ++j) {
                    double diff = points[i][j] - centroids[c][j];
                    dist += diff * diff;
                }
                if (dist < bestDist) {
                    bestDist = dist;
                    bestCluster = c;
                }
            }
            labels[i] = bestCluster;
        }

        // 2) Update step
        // Reset accumulators
        for (int c = 0; c < k; c++) {
            fill(newCentroids[c].begin(), newCentroids[c].end(), 0.0);
            counts[c] = 0;
        }
        // Sum points in each cluster
        for (int i = 0; i < n; ++i) {
            int c = labels[i];
            counts[c]++;
            for (int j = 0; j < d; ++j) {
                newCentroids[c][j] += points[i][j];
            }
        }
        // Compute means (centroids)
        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                for (int j = 0; j < d; ++j) {
                    newCentroids[c][j] /= counts[c];
                }
            } else {
                // if empty cluster, leave centroid unchanged
                newCentroids[c] = centroids[c];
            }
        }
        centroids.swap(newCentroids);
    }

    // Output centroids
    cout << "Centroids:\n";
    for (int c = 0; c < k; ++c) {
        for (int j = 0; j < d; ++j) {
            cout << centroids[c][j] << (j+1<d ? ' ' : '\n');
        }
    }

    // Output cluster assignments
    cout << "Cluster assignments:\n";
    for (int i = 0; i < n; ++i) {
        cout << labels[i] << (i+1<n ? ' ' : '\n');
    }

    return 0;
}

// // Train K-Means: returns final centroids and labels for each point
// function trainKMeans(points[n][d], k, I) -> (centroids[k][d], labels[n]):
//   // 1) Initialize centroids (e.g. first k points or random sample)
//   for c in 0..k-1:
//     centroids[c] = choose one point from points
//   // 2) Iterate assignment + update I times
//   for iter in 1..I:
//     // Assignment step
//     for i in 0..n-1:
//       labels[i] = argmin_c distance(points[i], centroids[c])
//     // Update step
//     for c in 0..k-1:
//       clusterPoints = [ points[i] for i in 0..n-1 if labels[i] == c ]
//       if clusterPoints not empty:
//         for j in 0..d-1:
//           centroids[c][j] = average( clusterPoints[*][j] )
//   return centroids, labels
//
// // Predict the cluster for a new query point
// function predictKMeans(centroids[k][d], query[d]) -> clusterIndex:
//   best = 0
//   bestDist = distance(query, centroids[0])
//   for c in 1..k-1:
//     dist = distance(query, centroids[c])
//     if dist < bestDist:
//       bestDist = dist
//       best = c
//   return best
