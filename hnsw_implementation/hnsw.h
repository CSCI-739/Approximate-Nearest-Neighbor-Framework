#ifndef HNSW_H
#define HNSW_H

#include <random>
#include <vector>
#include <unordered_map>
#include <iostream>

#include "../CPU/computations.h"
using namespace std;


struct HNSWGraph {
	HNSWGraph(int _M, int _maxNeighbours, int _maxNeighboursIn0, int _efficientConstruction, int _max_layers):M(_M),maxNeighbours(_maxNeighbours),maxNeighboursIn0(_maxNeighboursIn0),efficientConstruction(_efficientConstruction),max_layers(_max_layers){
		layerEdgeLists.push_back(unordered_map<int, vector<int>>());
	}
	int M;
	int maxNeighbours;
	int maxNeighboursIn0;
	int efficientConstruction;
	int max_layers;
	int itemNum;
	vector<Item> items;
	vector<unordered_map<int, vector<int>>> layerEdgeLists;
	int enterNode;

	default_random_engine generator;
	void addEdge(int st, int ed, int lc);
	vector<int> searchLayer(Item& q, int ep, int ef, int lc);
	void Insert(Item& q);
	vector<int> KNNSearch(Item& q, int K, int N);

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