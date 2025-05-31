// Copyright (c) 2025 Marcin Pajak

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include "dataset.h"
#include "config.h"

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

int main(int argc, char** argv){
    std::string root = argc > 1 ? argv[1] : "train";
    auto data = load_dataset(root);
    size_t N = data.size();
    std::cout << "Loaded " << N << " samples\n";

    // model weights
    Mat W1 = randn(HIDDEN, INPUT);
    Vec b1(HIDDEN,0);
    Mat W2 = randn(OUTPUT, HIDDEN);
    Vec b2(OUTPUT,0);

    Vec x(INPUT), h(HIDDEN), h_act(HIDDEN), y_hat(OUTPUT), grad_h(HIDDEN);

    auto tic=std::chrono::high_resolution_clock::now();

    for(int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        size_t idx = 0;
        float epoch_loss = 0.f;
        while(idx < N)
        {
            size_t end = std::min(idx+BATCH, N);
            // zero gradients
            Mat gW1(HIDDEN, Vec(INPUT, 0)), gW2(OUTPUT, Vec(HIDDEN, 0));
            Vec gb1(HIDDEN, 0), gb2(OUTPUT, 0);

            for(size_t i = idx; i < end; ++i)
            {
                const auto& s = data[i];
                const Vec& in = s.pixels;
                int label = s.label;

                // ===== forward =====
                mmv(W1, in, h, b1);                 // hidden pre-act
                for(int j = 0; j < HIDDEN; ++j)
                {
                    h_act[j] = std::max(0.f, h[j]);
                }
                mmv(W2, h_act, y_hat, b2);          // output logits
                softmax(y_hat);

                // loss = -log p[label]
                epoch_loss += -std::log(std::max(y_hat[label], 1e-8f));

                // ===== backward (manual) =====
                Vec delta2 = y_hat; delta2[label] -= 1.f;       // dL/dz2
                for(int j = 0; j < HIDDEN; ++j)
                {
                    for(int k = 0; k < OUTPUT; ++k) 
                    {
                        gW2[k][j] += delta2[k] * h_act[j];
                    }
                }
                for (int k = 0; k < OUTPUT; ++k)
                {
                    gb2[k]+=delta2[k];
                }

                // propagate to hidden
                std::fill(grad_h.begin(), grad_h.end(), 0);
                for(int j = 0; j < HIDDEN; ++j)
                {
                    for(int k = 0; k < OUTPUT; ++k) 
                    {
                        grad_h[j] += W2[k][j] * delta2[k];
                    }
                    grad_h[j] *= (h[j] > 0);                   // ReLU grad
                }
                // grads for W1, b1
                for(int j = 0; j < INPUT; ++j)
                {
                    for(int k = 0; k < HIDDEN; ++k)
                        {
                            gW1[k][j]+=grad_h[k]*in[j];
                        }
                }
                for(int k = 0; k < HIDDEN; ++k) 
                {
                    gb1[k] += grad_h[k];
                }
            }

            // SGD update (learning-rate / batch)
            float scale = LR / float(end-idx);
            for(int k = 0; k < HIDDEN; ++k)
            {
                b1[k] -= scale * gb1[k];
                for(int j = 0; j < INPUT; ++j) 
                {
                    W1[k][j] -= scale * gW1[k][j];
                }
            }
            for(int k = 0; k < OUTPUT; ++k)
            {
                b2[k] -= scale * gb2[k];
                for(int j = 0; j < HIDDEN; ++j) 
                {
                    W2[k][j] -= scale * gW2[k][j];
                }
            }

            idx = end;
        }
        std::cout << "Epoch " << epoch + 1 << " loss " << epoch_loss / N << "\n";
    }

    auto toc = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(toc - tic).count();
    std::cout << "CPU training time: " << secs << " s\n";
    
    return 0;
}
