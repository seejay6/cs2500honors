#include <bits/stdc++.h>
using namespace std;

/*
Genetic Algorithm (ga.cpp)

Variables:
  p            is the population size (number of candidate solutions)
  L            is the length of each bit-string genome
  g            is the number of generations to run
  mutationRate is the probability of flipping each bit during mutation

Algorithm steps (repeated for each generation):
  - Evaluate how good each bit-string is by counting its 1 bits.
  - Choose parents by holding tournaments: compare two candidates and pick the fitter.
  - Produce new children by splitting two parents at a random point and swapping tails.
  - Introduce variation by flipping bits in the children at random.
  - Form the next population from these new children.

Big-O analysis:
  - Fitness evaluation       is O(p * L)   via sum += population[i][j]
  - Tournament selection      is O(p)       via if (fitness[a] > fitness[b])
  - One-point crossover       is O(p * L)   via offspring[i][j] = parents[p1][j]
  - Mutation                  is O(p * L)   via population[i][j] = 1 - population[i][j]
  Total per generation: O(p * L)
  Total over g generations: O(g * p * L)

Goal:
  - This algorithm is an optimizer: it evolves candidate bit-strings to maximize the count of 1 bits and returns the best solution found. It does not predict new data.
*/


// ----------------------------------------------------------------------------
// evaluateFitness:
//   Returns vector of “number of 1s” for each genome.
//   O(p*L)
// ----------------------------------------------------------------------------
vector<int> evaluateFitness(const vector<vector<int>>& pop) {
    int p = pop.size(), L = pop[0].size();
    vector<int> fitness(p);
    for (int i = 0; i < p; ++i) {
        int sum = 0;
        for (int bit : pop[i]) sum += bit;
        fitness[i] = sum;
    }
    return fitness;
}

// ----------------------------------------------------------------------------
// tournamentSelection:
//   Select p parents via 1-vs-1 tournaments.
//   O(p)
// ----------------------------------------------------------------------------
vector<vector<int>> tournamentSelection(
    const vector<vector<int>>& pop,
    const vector<int>& fitness)
{
    int p = pop.size();
    vector<vector<int>> parents;
    parents.reserve(p);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, p - 1);

    for (int i = 0; i < p; ++i) {
        int a = dist(gen), b = dist(gen);
        // pick the one with higher fitness
        parents.push_back(
          (fitness[a] > fitness[b]) ? pop[a] : pop[b]
        );
    }
    return parents;
}

// ----------------------------------------------------------------------------
// onePointCrossover:
//   Create p offspring by picking two random parents and swapping
//   at a random crossover point.
//   O(p*L)
// ----------------------------------------------------------------------------
vector<vector<int>> onePointCrossover(const vector<vector<int>>& parents) {
    int p = parents.size(), L = parents[0].size();
    vector<vector<int>> offspring(p, vector<int>(L));

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> idxParent(0, p - 1);
    uniform_int_distribution<> idxCross(0, L - 1);

    for (int i = 0; i < p; ++i) {
        int p1 = idxParent(gen), p2 = idxParent(gen);
        int cp = idxCross(gen);
        // copy prefix from p1, suffix from p2
        for (int j = 0; j < cp; ++j)
            offspring[i][j] = parents[p1][j];
        for (int j = cp; j < L; ++j)
            offspring[i][j] = parents[p2][j];
    }
    return offspring;
}

// ----------------------------------------------------------------------------
// mutatePopulation:
//   Flip each bit with probability mutationRate.
//   O(p*L)
// ----------------------------------------------------------------------------
void mutatePopulation(vector<vector<int>>& pop, double mutationRate) {
    int p = pop.size(), L = pop[0].size();

    random_device rd;
    mt19937 gen(rd());
    bernoulli_distribution mutateDist(mutationRate);

    for (int i = 0; i < p; ++i)
        for (int j = 0; j < L; ++j)
            if (mutateDist(gen))
                pop[i][j] = 1 - pop[i][j];
}

// ----------------------------------------------------------------------------
// geneticAlgorithm:
//   Runs g generations starting from `population`, returns best genome.
//   O(g*p*L)
// ----------------------------------------------------------------------------
vector<int> geneticAlgorithm(
    vector<vector<int>> population,
    int g,
    double mutationRate)
{
    int p = population.size();
    vector<int> fitness;
    vector<vector<int>> parents, offspring;

    for (int gen = 0; gen < g; ++gen) {
        // 1) fitness
        fitness = evaluateFitness(population);
        // 2) selection
        parents = tournamentSelection(population, fitness);
        // 3) crossover
        offspring = onePointCrossover(parents);
        // 4) mutation
        mutatePopulation(offspring, mutationRate);
        // next gen
        population = move(offspring);
    }

    // final evaluation
    fitness = evaluateFitness(population);
    int bestIdx = max_element(fitness.begin(), fitness.end()) - fitness.begin();
    return population[bestIdx];
}

// ----------------------------------------------------------------------------
// main:
//   Reads input, runs GA, prints best individual and its fitness.
// ----------------------------------------------------------------------------
int main() {
    int p, L, g;
    double mutationRate;

    // read GA parameters
    if (!(cin >> p >> L >> g >> mutationRate)) {
        cerr << "Expected: p L g mutationRate\n";
        return 1;
    }

    // read initial population
    vector<vector<int>> population(p, vector<int>(L));
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < L; ++j) {
            cin >> population[i][j];
        }
    }

    // run GA
    vector<int> best = geneticAlgorithm(population, g, mutationRate);
    int bestFit = accumulate(best.begin(), best.end(), 0);

    // output
    cout << "Best individual: ";
    for (int bit : best) cout << bit;
    cout << "\nFitness: " << bestFit << "\n";
    return 0;
}


// // Returns the fitness (sum of bits) of one individual
// function evaluateFitness(individual[L]):
//   return sum(individual[0..L-1])
//
// // Evolves a population and returns the best chromosome
// function geneticAlgorithm(p, L, g, m, population[p][L]) -> bestIndividual[L]:
//   for gen in 1..g:
//     // 1) Evaluate everyone’s fitness
//     for i in 0..p-1:
//       fitness[i] = evaluateFitness(population[i])
//     // 2) Tournament selection
//     parents = []
//     repeat p times:
//       a, b = two random indices in [0,p)
//       parents.append( fitness[a] > fitness[b] ? population[a] : population[b] )
//     // 3) Crossover -> offspring
//     offspring = []
//     for i in 0..p-1:
//       A, B = two random parents
//       point = random integer in [1, L-1]
//       child = A[0..point-1] + B[point..L-1]
//       offspring.append(child)
//     // 4) Mutate each bit with probability m
//     for each child in offspring:
//       for j in 0..L-1:
//         if rand() < m: flip child[j]
//     population = offspring
//   // After g generations, pick the fittest
//   return argmax_i population[i] by evaluateFitness(population[i])
