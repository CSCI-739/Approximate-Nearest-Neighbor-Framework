#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include "hnsw.h"

namespace py = pybind11;

PYBIND11_MODULE(pyhnsw, m) {
    py::class_<Item>(m, "Item")
        .def(py::init<std::vector<double>>());

    py::class_<HNSWGraph>(m, "HNSWGraph")
        .def(py::init<int, int, int, int, int>())
        .def("addEdge", &HNSWGraph::addEdge)
        .def("searchLayer", &HNSWGraph::searchLayer)
        .def("Insert", &HNSWGraph::Insert)
        .def("KNNSearch", &HNSWGraph::KNNSearch)
        .def("printGraph", &HNSWGraph::printGraph);
}
