#ifndef HNSW_H
#define HNSW_H

#include <random>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <omp.h>

#include "../CPU/vector_initialize.h"
using namespace std;


struct HNSWGraph {
	HNSWGraph(int _M, int _MMax, int _MMax0, int _efConstruction, int _ml):M(_M),MMax(_MMax),MMax0(_MMax0),efConstruction(_efConstruction),ml(_ml){
		layerEdgeLists.push_back(unordered_map<int, vector<int>>());
	}
	int M;
	int MMax;
	int MMax0;
	int efConstruction;
	int ml;
	int itemNum;
	vector<Item> items;
	vector<unordered_map<int, vector<int>>> layerEdgeLists;
	int enterNode;

	default_random_engine generator;
	void addEdge(int st, int ed, int lc);
	vector<int> searchLayer(Item& q, int ep, int ef, int lc);
	void Insert(Item& q);
	vector<int> KNNSearch(Item& q, int K);

	void printGraph() {
		for (int l = 0; l < layerEdgeLists.size(); l++) {
			cout << "Layer:" << l << endl;
			for (auto it = layerEdgeLists[l].begin(); it != layerEdgeLists[l].end(); ++it) {
				cout << it->first << ":";
				for (auto ed: it->second) cout << ed << " ";
				cout << endl;
			}
		}
	}
};

#endif