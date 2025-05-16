#include <bits/stdc++.h>
using namespace std;

/*
Vanilla RNN Classifier (rnn.cpp)

Variables:
  n         is the number of training sequences
  T         is the length of each sequence
  D         is the dimensionality of input features per time step
  H         is the size of the hidden-state vector
  E         is the number of epochs for training
  lr        is the learning rate
  Wxh       is a T×H array for input-to-hidden weights
  Whh       is an H×H matrix for hidden-to-hidden weights
  Why       is a length-H vector for hidden-to-output weights
  bh        is a length-H vector of hidden biases
  by        is a scalar output bias
  sequences holds the n input sequences (each T×D)
  labels    holds the n target labels (+1 or -1)
  query     is one sequence of length T×D to classify

Algorithm steps (for each of E epochs):
  - Forward pass for each sequence:
      initialize hidden state h to zeros
      for each time t from 0 to T-1:
        combine input[t] and previous h via Wxh and Whh, add bh, apply tanh to get new h
      compute output from final h using Why and by
  - Backward pass (BPTT) for each sequence:
      compute gradient of loss at the output
      for t from T-1 down to 0:
        backpropagate through tanh and accumulate gradients for Wxh[t], Whh, and bh
      accumulate gradients for Why and by
  - Parameter update:
      subtract lr times each accumulated gradient from Wxh, Whh, Why, bh, and by

Big-O analysis:
  - Forward pass per sequence       is O(T * H * D + T * H * H) via operations like
      a += Wxh[t][h] * input[t][d]
      a += Whh[h][h2] * h_prev[h2]
  - Backward pass per sequence      is O(T * H * H) via operations like
      dWhh[h1][h2] += da[h1] * h_prev[h2]
  - Total over E epochs and n sequences is O(E * n * (T * H * D + T * H^2))

Goal:
  Train a recurrent neural network to classify sequences as +1 or -1 by learning weights via backpropagation through time.
*/


static double randUniform(double a, double b) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<> dist(a, b);
    return dist(gen);
}

int main(){
    int n, T, D, H, E;
    double lr;
    if(!(cin >> n >> T >> D >> H >> E >> lr)){
        cerr << "Usage: n T D H E lr\n";
        return 1;
    }

    // Read training data
    vector<vector<vector<double>>> X(n, vector<vector<double>>(T, vector<double>(D)));
    vector<double> Y(n);
    for(int i=0;i<n;++i){
        for(int t=0;t<T;++t){
            for(int d=0;d<D;++d){
                cin >> X[i][t][d];
            }
        }
        cin >> Y[i];
    }

    // Read query sequence
    vector<vector<double>> Xq(T, vector<double>(D));
    for(int t=0;t<T;++t)
        for(int d=0;d<D;++d)
            cin >> Xq[t][d];

    // Initialize parameters
    // Wxh: H×D, Whh: H×H, Why: 1×H, bh: H, by: scalar
    vector<vector<double>> Wxh(H, vector<double>(D)),
                         Whh(H, vector<double>(H));
    vector<double> Why(H), bh(H, 0.0);
    double by = 0.0;

    // random small weights
    for(int i=0;i<H;++i){
        for(int j=0;j<D;++j) Wxh[i][j] = randUniform(-0.1,0.1);
        for(int j=0;j<H;++j) Whh[i][j] = randUniform(-0.1,0.1);
        Why[i] = randUniform(-0.1,0.1);
    }

    // Training via BPTT
    vector<vector<double>> h(T+1, vector<double>(H));
    vector<vector<double>> a(T+1, vector<double>(H));
    for(int epoch=0; epoch<E; ++epoch){
        for(int i=0;i<n;++i){
            // Forward pass
            fill(h[0].begin(), h[0].end(), 0.0);
            for(int t=1; t<=T; ++t){
                for(int hh=0; hh<H; ++hh){
                    double sum = bh[hh];
                    // input->hidden
                    for(int d=0; d<D; ++d)
                        sum += Wxh[hh][d] * X[i][t-1][d];
                    // hidden->hidden
                    for(int pp=0; pp<H; ++pp)
                        sum += Whh[hh][pp] * h[t-1][pp];
                    a[t][hh] = sum;
                    h[t][hh] = tanh(sum);
                }
            }
            double y_pred = by;
            for(int hh=0; hh<H; ++hh)
                y_pred += Why[hh] * h[T][hh];

            double err = y_pred - Y[i];  // derivative of 0.5*(pred - y)^2

            // initialize gradients
            vector<vector<double>> dWxh(H, vector<double>(D,0.0)),
                                   dWhh(H, vector<double>(H,0.0));
            vector<double> dWhy(H,0.0), dbh(H,0.0);
            double dby = err;
            // output->hidden
            for(int hh=0; hh<H; ++hh)
                dWhy[hh] = err * h[T][hh];

            // backprop through time
            vector<double> dh(H);
            for(int hh=0; hh<H; ++hh)
                dh[hh] = err * Why[hh];
            for(int t=T; t>=1; --t){
                vector<double> da(H);
                for(int hh=0; hh<H; ++hh){
                    // derivative through tanh
                    da[hh] = dh[hh] * (1 - h[t][hh]*h[t][hh]);
                    dbh[hh] += da[hh];
                }
                // accumulate parameter gradients
                for(int hh=0; hh<H; ++hh){
                    for(int d=0; d<D; ++d)
                        dWxh[hh][d] += da[hh] * X[i][t-1][d];
                    for(int pp=0; pp<H; ++pp)
                        dWhh[hh][pp] += da[hh] * h[t-1][pp];
                }
                // propagate to previous hidden
                vector<double> dh_prev(H,0.0);
                for(int pp=0; pp<H; ++pp){
                    for(int hh=0; hh<H; ++hh){
                        dh_prev[pp] += Whh[hh][pp] * da[hh];
                    }
                }
                dh.swap(dh_prev);
            }

            // update parameters
            for(int hh=0; hh<H; ++hh){
                Why[hh] -= lr * dWhy[hh];
                bh[hh]  -= lr * dbh[hh];
                for(int d=0; d<D; ++d)
                    Wxh[hh][d] -= lr * dWxh[hh][d];
                for(int pp=0; pp<H; ++pp)
                    Whh[hh][pp] -= lr * dWhh[hh][pp];
            }
            by -= lr * dby;
        }
    }

    // Inference on query
    fill(h[0].begin(), h[0].end(), 0.0);
    for(int t=1; t<=T; ++t){
        for(int hh=0; hh<H; ++hh){
            double sum = bh[hh];
            for(int d=0; d<D; ++d)
                sum += Wxh[hh][d] * Xq[t-1][d];
            for(int pp=0; pp<H; ++pp)
                sum += Whh[hh][pp] * h[t-1][pp];
            h[t][hh] = tanh(sum);
        }
    }
    double yq = by;
    for(int hh=0; hh<H; ++hh)
        yq += Why[hh] * h[T][hh];

    int pred = (yq >= 0 ? 1 : -1);
    cout << "Predicted label: " << pred << "\n";

    return 0;
}

// // Train a simple recurrent network on sequences
// function trainRNN(seqs[n][T], labels[n], H, epochs, lr) -> (Wxh[T][H], Whh[H][H], Why[H], bh[H], by):
//   // 1) Initialize weight matrices and biases
//   for t in 0..T-1: for h in 0..H-1: Wxh[t][h] = random()
//   for h1 in 0..H-1: for h2 in 0..H-1: Whh[h1][h2] = random()
//   for h in 0..H-1: Why[h] = random()
//   for h in 0..H-1: bh[h] = 0
//   by = 0
//
//   // 2) Perform epochs of training
//   for epoch in 1..epochs:
//     for i in 0..n-1:
//       // -- Forward pass --
//       h[0] = zero vector of length H
//       for t in 0..T-1:
//         a[t] = dot(Wxh[t], seqs[i][t]) + dot(Whh, h[t]) + bh
//         h[t+1] = tanh(a[t])
//       yhat = dot(Why, h[T]) + by
//
//       // -- Compute loss gradient at output --
//       // for MSE: dY = 2*(yhat - labels[i])
//       dY = yhat - labels[i]
//
//       // -- Backward pass (BPTT) --
//       // Gradients initialization
//       dWhy = zero vector H; dby = 0
//       dWhh = zero matrix H×H; dWxh[t] = zero matrix for each t; dbh = zero vector H
//       dh_next = zero vector H
//
//       // Output-layer gradients
//       for h in 0..H-1: dWhy[h] += dY * h[T][h]
//       dby += dY
//       // Propagate into last hidden state
//       dh = Why * dY + dh_next
//
//       // Backpropagate through time
//       for t in T-1 down to 0:
//         da = dh * (1 - h[t+1]^2)    // tanh’(a) = 1 - tanh(a)^2
//         dbh += da
//         for h1 in 0..H-1: 
//           dWxh[t][h1] += da * seqs[i][t]
//         for h1 in 0..H-1: for h2 in 0..H-1:
//           dWhh[h1][h2] += da * h[t][h2]
//         // Prepare dh for next step
//         dh = transpose(Whh) * da
//
//       // -- Parameter updates --
//       for t in 0..T-1: for h in 0..H-1: Wxh[t][h] -= lr * dWxh[t][h]
//       for h1 in 0..H-1: for h2 in 0..H-1: Whh[h1][h2] -= lr * dWhh[h1][h2]
//       for h in 0..H-1: Why[h] -= lr * dWhy[h]
//       for h in 0..H-1: bh[h]   -= lr * dbh[h]
//       by   -= lr * dby
//   return (Wxh, Whh, Why, bh, by)
//
//
// // Predict a label for one sequence
// function predictRNN(Wxh[T][H], Whh[H][H], Why[H], bh[H], by, query[T]) -> label:
//   h[0] = zero vector H
//   for t in 0..T-1:
//     a = dot(Wxh[t], query[t]) + dot(Whh, h[t]) + bh
//     h[t+1] = tanh(a)
//   yhat = dot(Why, h[T]) + by
//   return (yhat >= 0 ? +1 : -1)
