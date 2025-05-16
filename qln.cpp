#include <bits/stdc++.h>
using namespace std;

/*
Tabular Q-Learning (qln.cpp)

Variables:
  S            is the number of states
  A            is the number of actions available in each state
  T            is the number of observed transitions
  alpha        is the learning rate (0 < alpha <= 1)
  gamma        is the discount factor (0 <= gamma < 1)
  Q            is a S×A table of action-value estimates, initialized to zero
  transitions  is a list of T records, each containing (s, a, r, s')

Algorithm steps (for each of the T transitions):
  - Read a transition (s, a, r, s′) from the list.
  - Find the maximum Q-value in the next state s′ by checking all A actions.
  - Compute the TD target = r + gamma * max_future_value.
  - Update Q[s][a] by moving it fractionally toward the target.

Big-O analysis:
  - Scanning A actions to find the maximum in state s′ is O(A)  
    via code like:  
      maxQ = Q[s′][0];  
      for (int a2 = 1; a2 < A; ++a2) if (Q[s′][a2] > maxQ) maxQ = Q[s′][a2];
  - Performing that update for each of T transitions gives O(T * A).
  - Extracting a policy afterward by scanning S states and A actions is O(S * A).

Goal:
  Learn the optimal action-value function Q[s][a] from experience so that the derived policy (choose action argmax_a Q[s][a]) maximizes expected cumulative discounted reward.
*/


int main() {
    int S, A, T;
    double alpha, gamma;
    if (!(cin >> S >> A >> T >> alpha >> gamma)) {
        cerr << "Usage: S A T alpha gamma\n"
             << "Then T lines: s a r s'\n";
        return 1;
    }

    // Initialize Q-table to zeros
    vector<vector<double>> Q(S, vector<double>(A, 0.0));

    // Q-learning updates
    for (int i = 0; i < T; ++i) {
        int s, a, s_next;
        double r;
        cin >> s >> a >> r >> s_next;
        // Find max Q in next state
        double max_q_next = *max_element(Q[s_next].begin(), Q[s_next].end());
        // Update rule
        Q[s][a] += alpha * (r + gamma * max_q_next - Q[s][a]);
    }

    // Output final Q-table
    cout << "Final Q-table:\n";
    cout << fixed << setprecision(4);
    for (int s = 0; s < S; ++s) {
        for (int a = 0; a < A; ++a) {
            cout << Q[s][a] << (a + 1 < A ? ' ' : '\n');
        }
    }

    // Derive and output policy: best action per state
    cout << "Derived policy:\n";
    for (int s = 0; s < S; ++s) {
        int best_a = max_element(Q[s].begin(), Q[s].end()) - Q[s].begin();
        cout << best_a << (s + 1 < S ? ' ' : '\n');
    }

    return 0;
}

// // Train Q-learning on a sequence of transitions
// function trainQLearning(S, A, alpha, gamma, transitions[T]) -> Q[S][A]:
//   // 1) Initialize Q-table to zeros
//   for s in 0..S-1:
//     for a in 0..A-1:
//       Q[s][a] = 0
//
//   // 2) Process each observed transition
//   for each (s, a, r, s2) in transitions:
//     // find the best next‐state value
//     maxQ = Q[s2][0]
//     for a2 in 1..A-1:
//       if Q[s2][a2] > maxQ:
//         maxQ = Q[s2][a2]
//     // update rule
//     Q[s][a] += alpha * ( r + gamma * maxQ - Q[s][a] )
//
//   return Q
//
// // Derive the greedy policy from the Q-table
// function extractPolicy(Q[S][A]) -> policy[S]:
//   for s in 0..S-1:
//     bestA = 0
//     for a in 1..A-1:
//       if Q[s][a] > Q[s][bestA]:
//         bestA = a
//     policy[s] = bestA
//   return policy
