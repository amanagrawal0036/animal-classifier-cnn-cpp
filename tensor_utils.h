#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include <iostream>
#include <cmath>

#include <vector>
#include <numeric>
#include <random>

#include <algorithm>
#include <stdexcept>

using namespace std;

using Tensor3D = vector<vector<vector<float>>>;
using Tensor2D = vector<vector<float>>;
using Tensor1D = vector<float>;

inline Tensor3D createTensor3D(size_t depth, size_t height, size_t width, float init_val = 0.0f) {
    return Tensor3D(depth, Tensor2D(height, Tensor1D(width, init_val)));
}

inline Tensor2D createTensor2D(size_t height, size_t width, float init_val = 0.0f) {
    return Tensor2D(height, Tensor1D(width, init_val));
}

inline float randomFloat(float min_val, float max_val) {
    static mt19937 gen(random_device{}());
    uniform_real_distribution<float> distrib(min_val, max_val);
    return distrib(gen);
}

inline void initializeTensor3D(Tensor3D& tensor, float min_val = -0.1f, float max_val = 0.1f) {
    for (auto& plane : tensor) {
        for (auto& row : plane) {
            for (float& val : row) {
                val = randomFloat(min_val, max_val);
            }
        }
    }
}

inline void initializeTensor2D(Tensor2D& tensor, float min_val = -0.1f, float max_val = 0.1f) {
    for (auto& row : tensor) {
        for (float& val : row) {
            val = randomFloat(min_val, max_val);
        }
    }
}

inline void initializeTensor1D(Tensor1D& tensor, float min_val = -0.1f, float max_val = 0.1f) {
    for (float& val : tensor) {
        val = randomFloat(min_val, max_val);
    }
}

inline size_t getDepth(const Tensor3D& t) { return t.size(); }
inline size_t getHeight(const Tensor3D& t) { return t.empty() ? 0 : t[0].size(); }
inline size_t getWidth(const Tensor3D& t) { return (t.empty() || t[0].empty()) ? 0 : t[0][0].size(); }
inline size_t getHeight(const Tensor2D& t) { return t.size(); }
inline size_t getWidth(const Tensor2D& t) { return t.empty() ? 0 : t[0].size(); }

#endif // TENSOR_UTILS_H
