#include "hnsw.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
using namespace std;

void readInputFromFile(const string& filename, int& D, int& N, int& M, vector<Item>& base, vector<Item>& queries) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Unable to open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    string firstLine;
    getline(infile, firstLine);

    istringstream iss(firstLine);
    if (!(iss >> D >> N >> M) || iss.rdbuf()->in_avail() > 0) {
        cerr << "Invalid format in the input file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    base.reserve(N);
    queries.reserve(M);

    for (int i = 0; i < N; ++i) {
        vector<double> temp(D);
        for (int j = 0; j < D; ++j) {
            if (!(infile >> temp[j])) {
                cerr << "Invalid format in the input file " << filename << " at line " << (i + 2) << endl;
                exit(EXIT_FAILURE);
            }
        }
        base.emplace_back(temp);
    }

    for (int i = 0; i < M; ++i) {
        vector<double> temp(D);
        for (int j = 0; j < D; ++j) {
            if (!(infile >> temp[j])) {
                cerr << "Invalid format in the input file " << filename << " at line " << (N + i + 2) << endl;
                exit(EXIT_FAILURE);
            }
        }
        queries.emplace_back(temp);
    }

    infile.close();
}


int main(int argc, char* argv[]) {

    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input_filename> <output_filename>" << endl;
        return EXIT_FAILURE;
    }

    string filename = argv[1];
    string outputFilename = argv[2];
    int K = 5;

    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Unable to open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    int N = 0, D = 0, M = 0;

    std::vector<Item> base, queries;

    readInputFromFile(filename, D, N, M, base, queries);

    HNSWGraph myHNSWGraph(10, 30, 30, 10, 2);

    for (int i = 0; i < N; ++i) {
        myHNSWGraph.Insert(base[i]);
    }

    double total_brute_force_time = 0.0;
    double total_hnsw_time = 0.0;

    int numHits = 0;
    ofstream outfile(outputFilename);

    if (!outfile.is_open()) {
        cerr << "Unable to open output file " << outputFilename << endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < M; ++i) {
        Item query = queries[i];
        clock_t begin_time = clock();
        vector<pair<double, int>> distPairs;
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            distPairs.emplace_back(query.dist(base[j]), j);
        }
        sort(distPairs.begin(), distPairs.end());
        total_brute_force_time += double(clock() - begin_time) / CLOCKS_PER_SEC;

        begin_time = clock();
        vector<int> knns = myHNSWGraph.KNNSearch(query, K);
        for (size_t idx = 0; idx < knns.size(); ++idx) {
            outfile << knns[idx];
            if (idx != knns.size() - 1) {
                outfile << " ";
            }
        }
        outfile << endl;
        total_hnsw_time += double(clock() - begin_time) / CLOCKS_PER_SEC;

        if (knns[0] == distPairs[0].second) numHits++;
    }
    outfile.close();
    cout << numHits << " " << total_brute_force_time / M << " " << total_hnsw_time / M << endl;

	return 0;
}