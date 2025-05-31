#include <iostream>
#include <random>
#include <vector>
#include <cstring>

#include "dataset.h"
#include "config.h"
#include <hip/hip_runtime.h>

#define HIP_CHECK(err)                                                  \
    if (err != hipSuccess) {                                            \
        std::cerr << "HIP error at line " << __LINE__ << ": "           \
                  << hipGetErrorString(err) << std::endl;               \
        std::exit(1);                                                   \
    }

/* ------------------------------------------------------------------ */
/*                    Simple row-major linear kernels                 */
/* ------------------------------------------------------------------ */

// Y = X·Wᵀ + b
__global__
void linear_forward(int M, int N, int K,
                    const float* __restrict__ X,   // M×K
                    const float* __restrict__ W,   // N×K
                    const float* __restrict__ b,   // N
                    float*       __restrict__ Y)   // M×N
{
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = b[col];
    for (int k = 0; k < K; ++k)
        acc += X[row * K + k] * W[col * K + k];
    Y[row * N + col] = acc;
}

// element-wise ReLU
__global__
void relu_kernel(float* v, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) v[idx] = v[idx] > 0.0f ? v[idx] : 0.0f;
}

// dW[row,col] = Σ_m dY[m,row] * X[m,col]   (outer product accumulation)
__global__
void grad_w(int N, int K,
            const float* dY, const float* X,
            float* dW, int M)          // M=batch, N=output, K=input
{
    int row = blockIdx.x;   // output neuron
    int col = threadIdx.x;  // input feature
    if (row >= N || col >= K) return;

    float sum = 0.0f;
    for (int m = 0; m < M; ++m)
        sum += dY[m * N + row] * X[m * K + col];
    dW[row * K + col] = sum;
}

// in-place weight update: W += alpha * dW
__global__
void saxpy(float* W, const float* dW, float alpha, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) W[idx] += alpha * dW[idx];
}

/* ------------------------------------------------------------------ */
/*                       Helper initialisation                         */
/* ------------------------------------------------------------------ */

inline void init_rand(std::vector<float>& host, float sigma = 0.05f)
{
    std::mt19937 gen(123);
    std::normal_distribution<float> dist(0.f, sigma);
    for (auto& v : host) v = dist(gen);
}

/* ------------------------------------------------------------------ */
/*                               main                                  */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv)
{
    std::string root = (argc > 1) ? argv[1] : "train";
    auto dataset = load_dataset(root);
    size_t Nall  = dataset.size();
    std::cout << "Loaded " << Nall << " samples\n";

    /* ---- host pinned buffers for the active batch ---- */
    float *h_X, *h_Y, *h_dY;
    HIP_CHECK(hipHostMalloc(&h_X,  BATCH * INPUT  * sizeof(float)));
    HIP_CHECK(hipHostMalloc(&h_Y,  BATCH * OUTPUT * sizeof(float)));
    HIP_CHECK(hipHostMalloc(&h_dY, BATCH * OUTPUT * sizeof(float)));

    /* ---- device buffers ---- */
    float *d_X, *d_H, *d_Y;
    float *d_W1, *d_b1, *d_W2, *d_b2;
    float *d_dW1, *d_dW2, *d_dH;
    HIP_CHECK(hipMalloc(&d_X,  BATCH * INPUT  * 4));
    HIP_CHECK(hipMalloc(&d_H,  BATCH * HIDDEN * 4));
    HIP_CHECK(hipMalloc(&d_Y,  BATCH * OUTPUT * 4));
    HIP_CHECK(hipMalloc(&d_W1, INPUT * HIDDEN * 4));
    HIP_CHECK(hipMalloc(&d_b1, HIDDEN * 4));
    HIP_CHECK(hipMalloc(&d_W2, HIDDEN * OUTPUT * 4));
    HIP_CHECK(hipMalloc(&d_b2, OUTPUT * 4));
    HIP_CHECK(hipMalloc(&d_dW1, INPUT * HIDDEN * 4));
    HIP_CHECK(hipMalloc(&d_dW2, HIDDEN * OUTPUT * 4));
    HIP_CHECK(hipMalloc(&d_dH,  BATCH * HIDDEN * 4));

    /* ---- initialise weights ---- */
    std::vector<float> tmp(INPUT * HIDDEN);
    init_rand(tmp); HIP_CHECK(hipMemcpy(d_W1, tmp.data(), tmp.size()*4, hipMemcpyHostToDevice));
    tmp.assign(HIDDEN * OUTPUT, 0); init_rand(tmp);
    HIP_CHECK(hipMemcpy(d_W2, tmp.data(), tmp.size()*4, hipMemcpyHostToDevice));
    std::vector<float> zeros(HIDDEN, 0);
    HIP_CHECK(hipMemcpy(d_b1, zeros.data(), zeros.size()*4, hipMemcpyHostToDevice));
    zeros.assign(OUTPUT, 0);
    HIP_CHECK(hipMemcpy(d_b2, zeros.data(), zeros.size()*4, hipMemcpyHostToDevice));

    /* ---- timing events ---- */
    hipEvent_t evStart, evStop;
    hipEventCreate(&evStart);
    hipEventCreate(&evStop);
    hipEventRecord(evStart);

    std::mt19937 rng(42);

    /* =====================   training loop   ===================== */
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::shuffle(dataset.begin(), dataset.end(), rng);
        size_t idx = 0;
        float epoch_loss = 0.0f;

        while (idx < Nall) {
            /* -------- current mini-batch size -------- */
            const int cur = static_cast<int>(std::min(static_cast<size_t>(BATCH), Nall - idx));

            /* -------- copy batch to pinned host -------- */
            for (int m = 0; m < cur; ++m)
                std::memcpy(&h_X[m * INPUT], dataset[idx + m].pixels.data(),
                            INPUT * sizeof(float));

            /* -------- host → device -------- */
            HIP_CHECK(hipMemcpy(d_X, h_X, cur * INPUT * 4, hipMemcpyHostToDevice));

            /* -------- forward pass -------- */
            hipLaunchKernelGGL(linear_forward, dim3(cur), dim3(HIDDEN), 0, 0,
                               cur, HIDDEN, INPUT, d_X, d_W1, d_b1, d_H);

            /* ReLU */
            int reluBlocks  = (cur * HIDDEN + 255) / 256;
            hipLaunchKernelGGL(relu_kernel, dim3(reluBlocks), dim3(256), 0, 0,
                               d_H, cur * HIDDEN);

            hipLaunchKernelGGL(linear_forward, dim3(cur), dim3(OUTPUT), 0, 0,
                               cur, OUTPUT, HIDDEN, d_H, d_W2, d_b2, d_Y);

            /* -------- device → host logits -------- */
            HIP_CHECK(hipMemcpy(h_Y, d_Y, cur * OUTPUT * 4, hipMemcpyDeviceToHost));

            /* -------- softmax & loss on host -------- */
            for (int m = 0; m < cur; ++m) {
                float* row = &h_Y[m * OUTPUT];
                float maxv = *std::max_element(row, row + OUTPUT);
                float sum  = 0.f;
                for (int k = 0; k < OUTPUT; ++k) {
                    row[k] = std::exp(row[k] - maxv);
                    sum   += row[k];
                }
                for (int k = 0; k < OUTPUT; ++k) row[k] /= sum;

                int lbl = dataset[idx + m].label;
                epoch_loss += -std::log(std::max(row[lbl], 1e-8f));

                for (int k = 0; k < OUTPUT; ++k)
                    h_dY[m * OUTPUT + k] = row[k] - (k == lbl);
            }

            /* -------- host grad → device -------- */
            HIP_CHECK(hipMemcpy(d_dW2, h_dY, cur * OUTPUT * 4, hipMemcpyHostToDevice));

            /* -------- backward W2 -------- */
            hipLaunchKernelGGL(grad_w, dim3(OUTPUT), dim3(HIDDEN), 0, 0,
                               OUTPUT, HIDDEN, d_dW2, d_H, d_dW2, cur);

            /* -------- SGD update W2 only (demo) -------- */
            int saxN = HIDDEN * OUTPUT;
            hipLaunchKernelGGL(saxpy, dim3((saxN + 255)/256), dim3(256), 0, 0,
                               d_W2, d_dW2, -LR / cur, saxN);

            idx += cur;
        }
        std::cout << "Epoch " << (epoch + 1)
                  << "  avg-loss " << epoch_loss / Nall << '\n';
    }

    /* ---- stop timer ---- */
    hipEventRecord(evStop);
    hipEventSynchronize(evStop);
    float msec = 0.0f;
    hipEventElapsedTime(&msec, evStart, evStop);
    std::cout << "GPU (HIP) training wall-time: " << msec/1000.0f << " s\n";

    hipEventDestroy(evStart);
    hipEventDestroy(evStop);
    hipFree(d_X); hipFree(d_H); hipFree(d_Y);
    hipFree(d_W1); hipFree(d_b1);
    hipFree(d_W2); hipFree(d_b2);
    hipFree(d_dW1); hipFree(d_dW2); hipFree(d_dH);
    hipHostFree(h_X); hipHostFree(h_Y); hipHostFree(h_dY);

    return 0;
}
