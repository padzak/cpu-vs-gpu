// Copyright (c) 2025 Marcin Pajak

#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

// Aliases
using Vec = std::vector<float>;
using Mat = std::vector<Vec>;        // row-major

// initialise matrix with N(0,0.05)
static Mat randn(size_t rows, size_t cols){
    std::mt19937 gen(123);
    std::normal_distribution<float> d(0.f, 0.05f);
    Mat m(rows, Vec(cols));
    for(auto& r : m) 
    {
        for(float& v : r) 
        {
            v = d(gen);
        }
    }

    return m;
}

// y = W·x + b   (rows=W rows)
inline void mmv(const Mat& W, const Vec& x, Vec& y, const Vec& b)
{
    size_t rows = W.size(), cols = W[0].size();
    for(size_t i = 0; i < rows; ++i)
    {
        float sum = b[i];
        const float* w = &W[i][0];
        for(size_t j = 0; j < cols; ++j) 
        {
            sum += w[j] * x[j];
        }
        
        y[i] = sum;
    }
}

// softmax in-place
inline void softmax(Vec& v){
    float m =* std::max_element(v.begin(), v.end());
    float sum = 0;
    for(float& z : v)
    { 
        z = std::exp(z - m); 
        sum += z; 
    }
    for(float& z : v) 
    {
        z /= sum;
    }
}
