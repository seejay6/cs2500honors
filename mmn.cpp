#include <bits/stdc++.h>
using namespace std;

/*
Minimax Tic-Tac-Toe (mmn.cpp)

Variables:
  board     is an array of 9 integers representing the game state (0 empty, +1 X, -1 O)
  player    is the current mover (+1 for X’s turn, -1 for O’s turn)
  bestMove  stores the index (0–8) of the optimal move found
  bestScore stores the score (+1 win, 0 draw, -1 loss) for that move

Algorithm steps:
  - Define minimax(board, player):
      • If the board is a terminal state, return its score.
      • Otherwise, for each index i from 0 to 8:
          - If board[i] is empty, place player’s mark there.
          - Recursively call minimax(board, -player) to get score.
          - Undo the move.
          - If player==+1, keep the maximum score; if player==-1, keep the minimum score.
      • Return the best score.
  - In main, call minimax on the initial board and player to determine bestMove and bestScore, then print them.

Big-O analysis:
  - Terminal check is O(1) via scanning the fixed 8 winning lines in checkWinner
  - Branching at each level is O(b) via for(i=0; i<9; ++i) if(board[i]==0)
  - Combining scores is O(1) via max or min comparison
  - Overall time grows like O(b^d), where b≤9 is the branching factor and d≤9 is move depth

Goal:
  Determine the move that guarantees the best outcome (win, draw, or avoid loss) assuming both players play optimally.
*/


// Board is length-9 vector of {0, +1, -1}
using Board = vector<int>;

// Move: chosen index and its minimax score
struct Move {
    int index;
    int score;
};

// checkWinner:
//   Returns +1 if X wins, -1 if O wins, 0 if draw, 2 if ongoing.
//   O(1) time.
int checkWinner(const Board& b) {
    static const int wins[8][3] = {
        {0,1,2},{3,4,5},{6,7,8},
        {0,3,6},{1,4,7},{2,5,8},
        {0,4,8},{2,4,6}
    };
    for (auto& w : wins) {
        int sum = b[w[0]] + b[w[1]] + b[w[2]];
        if (sum == 3)  return +1;
        if (sum == -3) return -1;
    }
    for (int i = 0; i < 9; ++i)
        if (b[i] == 0) return 2;  // game ongoing
    return 0;  // draw
}

// minimax:
//   Recursively find optimal move for 'player' (+1 or -1).
//   Returns Move{index, score}.
//   O(b^d) time.
Move minimax(Board& board, int player) {
    int result = checkWinner(board);
    if (result != 2) {
        // Terminal node
        return Move{-1, result};
    }

    vector<Move> moves;
    // Generate legal moves
    for (int i = 0; i < 9; ++i) {
        if (board[i] == 0) {
            board[i] = player;
            Move m;
            m.index = i;
            m.score = minimax(board, -player).score;
            moves.push_back(m);
            board[i] = 0;
        }
    }

    // Choose best move
    Move bestMove;
    if (player == 1) {
        bestMove.score = -2;
        for (auto& m : moves)
            if (m.score > bestMove.score)
                bestMove = m;
    } else {
        bestMove.score = +2;
        for (auto& m : moves)
            if (m.score < bestMove.score)
                bestMove = m;
    }
    return bestMove;
}

int main() {
    Board board(9);
    int player;
    // Read 9 board values then player
    for (int i = 0; i < 9; ++i) {
        if (!(cin >> board[i])) {
            cerr << "Expected 9 board values\n";
            return 1;
        }
    }
    if (!(cin >> player) || (player != 1 && player != -1)) {
        cerr << "Expected player (+1 or -1)\n";
        return 1;
    }

    Move best = minimax(board, player);
    cout << "Best move index: " << best.index << "\n";
    cout << "Score: " << best.score << "\n";
    return 0;
}

// // minimax(board[9], player) -> best score from this position
// function minimax(board[9], player) -> score:
//   result = checkWinner(board)    // +1 if X wins, -1 if O wins, 0 draw, 2 non-terminal
//   if result != 2:
//     return result
//   if player == MAX:              // MAX wants highest score
//     best = -inf
//     // try every possible move
//     for i in 0..8:
//       if board[i] == EMPTY:
//         board[i] = player
//         score = minimax(board, -player)
//         board[i] = EMPTY
//         best = max(best, score)
//     return best
//   else:                          // MIN (the opponent) wants lowest score
//     best = +inf
//     for i in 0..8:
//       if board[i] == EMPTY:
//         board[i] = player
//         score = minimax(board, -player)
//         board[i] = EMPTY
//         best = min(best, score)
//     return best
//
// // predictMove(board[9], player) → (bestMove, bestScore)
// function predictMove(board[9], player) -> (moveIndex, score):
//   if player == MAX: bestScore = -inf else bestScore = +inf
//   bestMove = -1
//   for i in 0..8:
//     if board[i] == EMPTY:
//       board[i] = player
//       score = minimax(board, -player)
//       board[i] = EMPTY
//       if (player == MAX and score > bestScore) or
//          (player == MIN and score < bestScore):
//         bestScore = score
//         bestMove  = i
//   return (bestMove, bestScore)
