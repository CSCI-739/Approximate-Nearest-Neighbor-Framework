#include "hnsw.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

void randomTest(int numItems, int dim, int numQueries, int K) {
	std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::vector<Item> randomItems;
    randomItems.reserve(numItems);

    for (int i = 0; i < numItems; i++) {
        std::vector<double> temp(dim);
        for (int d = 0; d < dim; d++) {
            temp[d] = distribution(generator);
        }
        randomItems.emplace_back(temp);
    }

    std::shuffle(randomItems.begin(), randomItems.end(), generator);
	HNSWGraph myHNSWGraph(10, 30, 30, 10, 2);
	for (int i = 0; i < numItems; i++) {
		if (i % 10000 == 0) cout << i << endl;
		myHNSWGraph.Insert(randomItems[i]);
	}
	
	double total_brute_force_time = 0.0;
	double total_hnsw_time = 0.0;

	cout << "START QUERY" << endl;
	int numHits = 0;
	for (int i = 0; i < numQueries; i++) {
		vector<double> temp(dim);
		for (int d = 0; d < dim; d++) {
			temp[d] = distribution(generator);
		}
		Item query(temp);

		// Brute force
		clock_t begin_time = clock();
		vector<pair<double, int>> distPairs;
		for (int j = 0; j < numItems; j++) {
			if (j == i) continue;
			distPairs.emplace_back(query.dist(randomItems[j]), j);
		}
		sort(distPairs.begin(), distPairs.end());
		total_brute_force_time += double( clock () - begin_time ) /  CLOCKS_PER_SEC;

		begin_time = clock();
		vector<int> knns = myHNSWGraph.KNNSearch(query, K);
		// cout << "Printing vectors";
		// std::cout << "Contents of knns vector:" << std::endl;
		// for (size_t i = 0; i < knns.size(); ++i) {
		// 	std::cout << "knns[" << i << "] = " << knns[i] << std::endl;
		// }
		// cout << "\nPrinted Vectors";
		total_hnsw_time += double( clock () - begin_time ) /  CLOCKS_PER_SEC;

		if (knns[0] == distPairs[0].second) numHits++;
	}
	cout << numHits << " " << total_brute_force_time / numQueries  << " " << total_hnsw_time / numQueries << endl;
}

int main() {
	randomTest(10000, 4, 100, 5);
	return 0;
}