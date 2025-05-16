#include <bits/stdc++.h>
using namespace std;

/*
Decision Tree (ID3) (id3.cpp)

Variables:
  n        is the number of training samples
  m        is the number of binary features per sample
  X        is an n×m matrix of feature values (0 or 1)
  y        is a vector of length n of labels (0 or 1)
  features is a list of remaining feature indices to split on
  tree     is the root pointer of the constructed decision tree

Algorithm steps:
  - If all labels in the current subset are identical or no features remain, create a leaf node.
  - Otherwise compute the entropy of the labels in the current subset.
  - For each feature in the remaining list, split the samples into two groups (value 0 or 1), compute each group’s entropy, and calculate the information gain.
  - Select the feature with the highest gain and remove it from the feature list.
  - Partition the samples into left and right subsets based on that feature and recursively build the left and right child nodes.
  - Return a decision node storing the chosen feature and pointers to its two children.
  - To predict, start at the root and at each node test the query’s value on that node’s feature; follow left or right until reaching a leaf and return its label.

Big-O analysis:
  - Checking for pure labels        is O(n)     via all_of(y.begin(), y.end(), …)
  - Computing information gains     is O(m * n) via for each feature f: for i in 0..n-1 check X[i][f] and count
  - Recursing m levels              is O(m)     combining m iterations of the above
  Total training time               is O(m * (m * n)) = O(m^2 * n)
  - Prediction for one query        is O(m)     via following at most m tests in the tree

Goal:
  - Build a binary tree that perfectly splits the training data by selecting features that maximize information gain, then classify a new example by traversing that tree to a leaf.
*/


// Holds a vector of binary features and a 0/1 label
struct DataPoint {
    vector<int> features;  // m binary features
    int label;             // class label (0 or 1)
};

// Tree node: either an internal split or a leaf with a label
struct Node {
    bool isLeaf;
    int label;          // valid if isLeaf==true
    int featureIndex;   // which feature to test
    Node* left;         // branch for feature==0
    Node* right;        // branch for feature==1
    Node(): isLeaf(false), label(-1),
            featureIndex(-1), left(nullptr), right(nullptr) {}
};

// Compute −sumation p i*log2(p i) over labels {0,1}; O(n)
double entropy(const vector<DataPoint>& data) {
    int n = data.size();
    if (n == 0) return 0.0;
    int count0 = 0;
    for (const auto& pt : data)
        if (pt.label == 0) count0++;
    int count1 = n - count0;
    double p0 = double(count0) / n;
    double p1 = double(count1) / n;
    double ent = 0.0;
    if (p0 > 0) ent -= p0 * log2(p0);
    if (p1 > 0) ent -= p1 * log2(p1);
    return ent;
}

// Compute information gain for splitting on featureIndex; O(n)
double infoGain(const vector<DataPoint>& data, int featureIndex) {
    double baseEnt = entropy(data);
    vector<DataPoint> left, right;
    left.reserve(data.size());
    right.reserve(data.size());
    for (const auto& pt : data) {
        if (pt.features[featureIndex] == 0) left.push_back(pt);
        else                                 right.push_back(pt);
    }
    double pL = double(left.size()) / data.size();
    double pR = double(right.size()) / data.size();
    return baseEnt
         - (pL * entropy(left) + pR * entropy(right));
}

// Recursively build the decision tree; O(m^2*n)
Node* buildTree(const vector<DataPoint>& data,
                const vector<int>& features)
{
    Node* node = new Node();
    int n = data.size();
    if (n == 0) return node;

    // Check if all labels are the same
    int count0 = 0;
    for (const auto& pt : data)
        if (pt.label == 0) count0++;
    if (count0 == n || count0 == 0 || features.empty()) {
        node->isLeaf = true;
        node->label = (count0 > n/2 ? 0 : 1);
        return node;
    }

    // Find best feature by information gain
    int bestFeat = features[0];
    double bestGain = infoGain(data, bestFeat);
    for (int f : features) {
        double gain = infoGain(data, f);
        if (gain > bestGain) {
            bestGain = gain;
            bestFeat = f;
        }
    }
    // If no positive gain, make leaf with majority
    if (bestGain <= 0) {
        node->isLeaf = true;
        node->label = (count0 > n/2 ? 0 : 1);
        return node;
    }

    // Split data on bestFeat
    vector<DataPoint> left, right;
    left.reserve(n);
    right.reserve(n);
    for (const auto& pt : data) {
        if (pt.features[bestFeat] == 0) left.push_back(pt);
        else                             right.push_back(pt);
    }

    // Prepare remaining features (m−1)
    vector<int> rem;
    rem.reserve(features.size()-1);
    for (int f : features)
        if (f != bestFeat) rem.push_back(f);

    node->featureIndex = bestFeat;
    node->left  = buildTree(left,  rem);
    node->right = buildTree(right, rem);
    return node;
}

// Traverse the tree to predict one sample; O(m)
int predict(Node* node, const vector<int>& sample) {
    if (node->isLeaf) return node->label;
    int val = sample[node->featureIndex];
    if (val == 0)  return predict(node->left,  sample);
    else           return predict(node->right, sample);
}

int main() {
    int n, m;
    if (!(cin >> n >> m)) {
        cerr << "Usage: n m\n"
             << "Then n lines: m features (0/1) and 1 label (0/1)\n"
             << "Then 1 line: m features (0/1) for query\n";
        return 1;
    }

    vector<DataPoint> dataset(n);
    for (int i = 0; i < n; ++i) {
        dataset[i].features.resize(m);
        for (int j = 0; j < m; ++j)
            cin >> dataset[i].features[j];
        cin >> dataset[i].label;
    }

    vector<int> query(m);
    for (int j = 0; j < m; ++j)
        cin >> query[j];

    // Feature indices 0..m-1
    vector<int> features(m);
    iota(features.begin(), features.end(), 0);

    Node* root = buildTree(dataset, features);
    int label = predict(root, query);

    cout << "Predicted label: " << label << "\n";
    return 0;
}

// // Build the binary decision tree
// function buildTree(X[n][m], y[n], featuresList) -> TreeNode:
//   if all y[i] are identical:
//     return Leaf(label = y[0])
//   if featuresList is empty:
//     return Leaf(label = majorityLabel(y))
//   baseH = entropy(y)
//   bestGain = 0; bestFeat = null
//   for each f in featuresList:
//     (y0, y1) = partition y by X[*][f] == 0 or 1
//     gain = baseH - (|y0|/n)*entropy(y0) - (|y1|/n)*entropy(y1)
//     if gain > bestGain:
//       bestGain = gain; bestFeat = f
//   if bestFeat is null:
//     return Leaf(label = majorityLabel(y))
//   remove bestFeat from featuresList -> remFeatures
//   (X0, y0) = samples with X[*][bestFeat] == 0
//   (X1, y1) = samples with X[*][bestFeat] == 1
//   left  = buildTree(X0, y0, remFeatures)
//   right = buildTree(X1, y1, remFeatures)
//   return Node(test = bestFeat, left, right)
//
// // Predict with the built tree
// function predictDT(tree, query[m]) -> label:
//   if tree is Leaf: 
//     return tree.label
//   if query[tree.test] == 0:
//     return predictDT(tree.left, query)
//   else:
//     return predictDT(tree.right, query)
