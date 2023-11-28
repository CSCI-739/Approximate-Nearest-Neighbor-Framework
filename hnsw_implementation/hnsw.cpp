#include "hnsw.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>
#include <omp.h>
using namespace std;

vector<int> HNSWGraph::searchLayer(Item& q, int entry_point, int ef, int lc) {
	set<pair<double, int>> candidates;
	set<pair<double, int>> nearestNeighbors;
	unordered_set<int> isVisited;

	double td = q.dist(items[entry_point]);
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
		for (int ed: layerEdgeLists[lc][nid]) {
			if (isVisited.find(ed) != isVisited.end()) {
				continue;
			}
			fi = nearestNeighbors.end(); 
			fi--;
			isVisited.insert(ed);
			td = q.dist(items[ed]);
			if ((td < fi->first) || nearestNeighbors.size() < ef) {
				candidates.insert(make_pair(td, ed));
				nearestNeighbors.insert(make_pair(td, ed));
				if (nearestNeighbors.size() > ef) {
					nearestNeighbors.erase(fi);
				}
			}
		}
	}
	vector<int> results;
	for(auto &p: nearestNeighbors) results.push_back(p.second);
	return results;
}

vector<int> HNSWGraph::KNNSearch(Item& q, int K) {
	int maxLyer = layerEdgeLists.size() - 1;
	int entry_point = enterNode;
	for (int l = maxLyer; l >= 1; l--) {
		entry_point = searchLayer(q, entry_point, 1, l)[0];
	}
	return searchLayer(q, entry_point, K, 0);
}

void HNSWGraph::addEdge(int st, int ed, int lc) {
	if (st == ed) {
		return;
	}
	layerEdgeLists[lc][st].push_back(ed);
	layerEdgeLists[lc][ed].push_back(st);
}

void HNSWGraph::Insert(Item& q) {
	int nid = items.size();
	itemNum++; items.push_back(q);
	int maxLyer = layerEdgeLists.size() - 1;
	int l = 0;
	uniform_real_distribution<double> distribution(0.0,1.0);
	while(l < ml && (1.0 / ml <= distribution(generator))) {
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
	for (int i = maxLyer; i > l; i--) {
		entry_point = searchLayer(q, entry_point, 1, i)[0];
	}
	#pragma omp parallel for
	for (int i = min(l, maxLyer); i >= 0; i--) {
		int MM = l == 0 ? MMax0 : MMax;
		vector<int> neighbors = searchLayer(q, entry_point, efConstruction, i);
		vector<int> selectedNeighbors = vector<int>(neighbors.begin(), neighbors.begin()+min(int(neighbors.size()), M));
		for (int n: selectedNeighbors) {
			addEdge(n, nid, i);
		}
		for (int n: selectedNeighbors) {
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
		entry_point = selectedNeighbors[0];
	}
	if (l == layerEdgeLists.size() - 1) {
		enterNode = nid;
	}
}