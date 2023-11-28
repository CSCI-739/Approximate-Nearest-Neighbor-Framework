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

void readInputFromFile(const string& filename, int& dim, int& numItems, int& numQueries, vector<Item>& randomItems, vector<Item>& queries) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Unable to open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    string firstLine;
    getline(infile, firstLine);

    istringstream iss(firstLine);
    if (!(iss >> dim >> numItems >> numQueries) || iss.rdbuf()->in_avail() > 0) {
        cerr << "Invalid format in the input file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    randomItems.reserve(numItems);
    queries.reserve(numQueries);

    // Read base vectors
    for (int i = 0; i < numItems; ++i) {
        vector<double> temp(dim);
        for (int j = 0; j < dim; ++j) {
            if (!(infile >> temp[j])) {
                cerr << "Invalid format in the input file " << filename << " at line " << (i + 2) << endl;
                exit(EXIT_FAILURE);
            }
        }
        randomItems.emplace_back(temp);
    }

    // Read queries
    for (int i = 0; i < numQueries; ++i) {
        vector<double> temp(dim);
        for (int j = 0; j < dim; ++j) {
            if (!(infile >> temp[j])) {
                cerr << "Invalid format in the input file " << filename << " at line " << (numItems + i + 2) << endl;
                exit(EXIT_FAILURE);
            }
        }
        queries.emplace_back(temp);
    }

    infile.close();
}


int main(int argc, char* argv[]) {

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <filename>" << endl;
        return EXIT_FAILURE;
    }

    string filename = argv[1];
    int K = 5;

    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Unable to open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    int numItems = 0, dim = 0, numQueries = 0;

    std::vector<Item> randomItems, queries;

    readInputFromFile(filename, dim, numItems, numQueries, randomItems, queries);

    HNSWGraph myHNSWGraph(10, 30, 30, 10, 2);

    for (int i = 0; i < numItems; ++i) {
        myHNSWGraph.Insert(randomItems[i]);
    }

    double total_brute_force_time = 0.0;
    double total_hnsw_time = 0.0;

    int numHits = 0;

    for (int i = 0; i < numQueries; ++i) {
        Item query = queries[i];

        // Brute force
        clock_t begin_time = clock();
        vector<pair<double, int>> distPairs;
        for (int j = 0; j < numItems; ++j) {
            if (j == i) continue;
            distPairs.emplace_back(query.dist(randomItems[j]), j);
        }
        sort(distPairs.begin(), distPairs.end());
        total_brute_force_time += double(clock() - begin_time) / CLOCKS_PER_SEC;

        begin_time = clock();
        vector<int> knns = myHNSWGraph.KNNSearch(query, K);
        total_hnsw_time += double(clock() - begin_time) / CLOCKS_PER_SEC;

        if (knns[0] == distPairs[0].second) numHits++;
    }
    cout << numHits << " " << total_brute_force_time / numQueries << " " << total_hnsw_time / numQueries << endl;

	return 0;
}