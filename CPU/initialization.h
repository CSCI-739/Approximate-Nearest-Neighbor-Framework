#ifndef ITEMS_H
#define ITEMS_H

#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <memory>
#include <string>
#include <sstream>
#include "omp.h"
using namespace std;

struct Item {
    Item(std::vector<double> _values) : values(_values) {}
    std::vector<double> values;

    double cosine_similarity(Item& other) {
        double dot_product = 0.0;
        double magnitude_this = 0.0;
        double magnitude_other = 0.0;

        for (size_t i = 0; i < values.size(); ++i) {
            dot_product += values[i] * other.values[i];
            magnitude_this += values[i] * values[i];
            magnitude_other += other.values[i] * other.values[i];
        }

        magnitude_this = sqrt(magnitude_this);
        magnitude_other = sqrt(magnitude_other);

        if (magnitude_this == 0 || magnitude_other == 0) {
            return 0.0; 
        }

        return dot_product / (magnitude_this * magnitude_other);
    }

    double dist(Item& other) {
        double result = 0.0;
        for (size_t i = 0; i < values.size(); i++) {
            result += (values[i] - other.values[i]) * (values[i] - other.values[i]);
        }
        return result;
    }

    void normalize() {
        double sum = 0.0;
        for (double val : values) {
            sum += val * val;
        }

        double magnitude = std::sqrt(sum);
        if (magnitude > 0.0) {
            for (double& val : values) {
                val /= magnitude;
            }
        }
    }

    double cosine_similarity_with_normalisation(Item& other) {
        double dot_product = 0.0;
        for (size_t i = 0; i < values.size(); ++i) {
            dot_product += values[i] * other.values[i];
        }

        return dot_product;
    }
};

#endif 
