#include "hnsw.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>
#include <omp.h>
#include <cmath> 
#include <limits> 
using namespace std;

vector<int> HNSWGraph::searchLayer(Item& query, int entry_point, int efficient, int layers) {
	if (layers < 0 || layers >= layerEdgeLists.size()) {
        cerr << "Error: Invalid layer index." << endl;
        exit(EXIT_FAILURE); 
    }
	set<pair<double, int>> candidates;
	set<pair<double, int>> nearestNeighbors;
	unordered_set<int> isVisited;

	double td = query.dist(items[entry_point]);
	candidates.insert(make_pair(td, entry_point));
	nearestNeighbors.insert(make_pair(td, entry_point));
	isVisited.insert(entry_point);
	while (!candidates.empty()) {
		auto ci = candidates.begin(); 
		candidates.erase(candidates.begin());
		int nid = ci->second;
		auto fi = nearestNeighbors.end(); fi--;
		if (ci->first > fi->first) {
			break;
		}
		for (int ed: layerEdgeLists[layers][nid]) {
			if (isVisited.find(ed) != isVisited.end()) {
				continue;
			}
			fi = nearestNeighbors.end(); 
			fi--;
			isVisited.insert(ed);
			td = query.dist(items[ed]);
			if ((td < fi->first) || nearestNeighbors.size() < efficient) {
				candidates.insert(make_pair(td, ed));
				nearestNeighbors.insert(make_pair(td, ed));
				if (nearestNeighbors.size() > efficient) {
					nearestNeighbors.erase(fi);
				}
			}
		}
	}
	vector<int> results;
	for(auto &p: nearestNeighbors) results.push_back(p.second);
	return results;
}

vector<int> HNSWGraph::KNNSearch(Item& query, int K, int N) {

	if (K <= 0 || std::ceil(K) != K || K > std::numeric_limits<int>::max()) {
		cerr << "Error: Invalid value of K for KNNSearch." << endl;
		exit(EXIT_FAILURE); 
	}

	if (K > N) {
        std::cerr << "Error: Value of K (" << K << ") is greater than the number of data points (N = " << N << ")." << std::endl;
        exit(EXIT_FAILURE);
    }
	
	int maxLayer = layerEdgeLists.size() - 1;
	int entry_point = enterNode;
	for (int l = maxLayer; l >= 1; l--) {
		entry_point = searchLayer(query, entry_point, 1, l)[0];
	}
	return searchLayer(query, entry_point, K, 0);
}

void HNSWGraph::addEdge(int st, int ed, int layers) {
	if (layers < 0 || layers >= layerEdgeLists.size()) {
        cerr << "Error: Invalid layer index." << endl;
        return;
    }
	if (st == ed) {
		return;
	}
	layerEdgeLists[layers][st].push_back(ed);
	layerEdgeLists[layers][ed].push_back(st);
}

void HNSWGraph::Insert(Item& query) {
	int nid = items.size();
	itemNum++; 
	items.push_back(query);
	int maxLayer = layerEdgeLists.size() - 1;
	int l = 0;
	uniform_real_distribution<double> distribution(0.0,1.0);
	while(l < max_layers && (1.0 / max_layers <= distribution(generator))) {
		l++;
		if (layerEdgeLists.size() <= l) {
			layerEdgeLists.push_back(unordered_map<int, vector<int>>());
		}
	}
	if (nid == 0) {
		enterNode = nid;
		return;
	}
	int entry_point = enterNode;
	for (int i = maxLayer; i > l; i--) {
		entry_point = searchLayer(query, entry_point, 1, i)[0];
	}
	#pragma omp parallel for
	for (int i = min(l, maxLayer); i >= 0; i--) {
		int MM = l == 0 ? maxNeighboursIn0 : maxNeighbours;
		vector<int> neighbours = searchLayer(query, entry_point, efficientConstruction, i);
		vector<int> selectedNeighbours = vector<int>(neighbours.begin(), neighbours.begin()+min(int(neighbours.size()), M));
		for (size_t j = 0; j < selectedNeighbours.size(); j++) {
            int n = selectedNeighbours[j];
            addEdge(n, nid, i);
        }
		for (size_t j = 0; j < selectedNeighbours.size(); j++)  {
			int n = selectedNeighbours[j];
			if (layerEdgeLists[i][n].size() > MM) {
				vector<pair<double, int>> distPairs;
				for (int nn: layerEdgeLists[i][n]) {
					distPairs.emplace_back(items[n].dist(items[nn]), nn);
				}
				sort(distPairs.begin(), distPairs.end());
				layerEdgeLists[i][n].clear();
				for (int d = 0; d < min(int(distPairs.size()), MM); d++) {
					layerEdgeLists[i][n].push_back(distPairs[d].second);
				}
			}
		}
		entry_point = selectedNeighbours[0];
	}
	if (l == layerEdgeLists.size() - 1) {
		enterNode = nid;
	}
}