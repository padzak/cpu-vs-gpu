// gpu_net.cu
#include "dataset.h"
#include "config.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(err) if(err){ \
    std::cerr<<"CUDA "<<__LINE__<<" "<<cudaGetErrorString(err)<<"\n"; exit(1); }

__global__
void linear_forward(int M,int N,int K,
                    const float* __restrict__ X,   // M×K
                    const float* __restrict__ W,   // N×K
                    const float* __restrict__ b,   // N
                    float*       __restrict__ Y)   // M×N
{
    int row = blockIdx.x;
    int col = threadIdx.x;
    if(row>=M || col>=N) return;
    float acc = b[col];
    for(int k=0;k<K;++k)
        acc += X[row*K+k] * W[col*K+k];
    Y[row*N+col] = acc;
}

// <<<N blocks, K threads>>> accumulate dW[k][j] += dY[k]*X[j]
__global__
void grad_w(int N,int K,
            const float* dY, const float* X,
            float* dW, int M)
{
    int row = blockIdx.x;   // output unit
    int col = threadIdx.x;  // input dim
    if(row>=N || col>=K) return;
    float sum=0;
    for(int m=0;m<M;++m) sum += dY[m*N+row]*X[m*K+col];
    dW[row*K+col]=sum;
}

inline void init_rand(float* ptr,size_t n,float std=0.05f){
    std::mt19937 gen(123);
    std::normal_distribution<float> d(0.f,std);
    for(size_t i=0;i<n;++i) ptr[i]=d(gen);
}

int main(int argc,char** argv){
    std::string root=argc>1?argv[1]:"train";
    auto data = load_dataset(root);
    size_t Nall=data.size();
    std::cout<<"Loaded "<<Nall<<" samples\n";

    // ----- allocate host pinned memory for a mini-batch -----
    float *h_X,*h_Y,*h_dY;
    CUDA_CHECK(cudaMallocHost(&h_X, BATCH*INPUT*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_Y, BATCH*OUTPUT*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_dY,BATCH*OUTPUT*sizeof(float)));

    // ----- allocate device tensors -----
    float *d_X,*d_W1,*d_b1,*d_H,*d_W2,*d_b2,*d_Y,*d_dW1,*d_dW2,*d_dH;
    CUDA_CHECK(cudaMalloc(&d_X ,BATCH*INPUT*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H ,BATCH*HIDDEN*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y ,BATCH*OUTPUT*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1,INPUT*HIDDEN*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1,HIDDEN*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2,HIDDEN*OUTPUT*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2,OUTPUT*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW1,INPUT*HIDDEN*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW2,HIDDEN*OUTPUT*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dH ,BATCH*HIDDEN*sizeof(float)));

    // ----- init weights -----
    std::vector<float> tmp(INPUT*HIDDEN); init_rand(tmp.data(),tmp.size()); CUDA_CHECK(cudaMemcpy(d_W1,tmp.data(),tmp.size()*4,cudaMemcpyHostToDevice));
    tmp.resize(HIDDEN*OUTPUT); init_rand(tmp.data(),tmp.size()); CUDA_CHECK(cudaMemcpy(d_W2,tmp.data(),tmp.size()*4,cudaMemcpyHostToDevice));
    tmp.resize(HIDDEN,0);                                     CUDA_CHECK(cudaMemcpy(d_b1,tmp.data(),tmp.size()*4,cudaMemcpyHostToDevice));
    tmp.resize(OUTPUT,0);                                     CUDA_CHECK(cudaMemcpy(d_b2,tmp.data(),tmp.size()*4,cudaMemcpyHostToDevice));

    cudaEvent_t start,stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    std::mt19937 rng(42);
    for(int epoch=0;epoch<EPOCHS;++epoch){
        std::shuffle(data.begin(),data.end(),rng);
        size_t idx=0;
        while(idx<Nall){
            size_t cur=std::min((size_t)BATCH,Nall-idx);
            // copy pictures to pinned host buffer
            for(size_t m=0;m<cur;++m)
                std::memcpy(&h_X[m*INPUT],data[idx+m].pixels.data(),INPUT*sizeof(float));
            CUDA_CHECK(cudaMemcpy(d_X,h_X,cur*INPUT*4,cudaMemcpyHostToDevice));

            // ---- forward hidden ----
            linear_forward<<<cur, HIDDEN>>>(cur,HIDDEN,INPUT,d_X,d_W1,d_b1,d_H);
            // ReLU
            int threads=256, blocks=(cur*HIDDEN+threads-1)/threads;
            relu_kernel<<<blocks,threads>>>(d_H,cur*HIDDEN);
            // forward output
            linear_forward<<<cur, OUTPUT>>>(cur,OUTPUT,HIDDEN,d_H,d_W2,d_b2,d_Y);

            // ---- softmax + loss grad on host (small) ----
            CUDA_CHECK(cudaMemcpy(h_Y,d_Y,cur*OUTPUT*4,cudaMemcpyDeviceToHost));
            float loss=0;
            for(size_t m=0;m<cur;++m){
                // softmax
                float* row=&h_Y[m*OUTPUT];
                float maxv=*std::max_element(row,row+OUTPUT);
                float sum=0; for(int k=0;k<OUTPUT;++k){ row[k]=std::exp(row[k]-maxv); sum+=row[k]; }
                for(int k=0;k<OUTPUT;++k) row[k]/=sum;
                int lbl=data[idx+m].label;
                loss += -std::log(std::max(row[lbl],1e-8f));
                // dY = prob - 1(label)
                for(int k=0;k<OUTPUT;++k) h_dY[m*OUTPUT+k]=row[k] - (k==lbl);
            }
            CUDA_CHECK(cudaMemcpy(d_dW2,h_dY,cur*OUTPUT*4,cudaMemcpyHostToDevice));

            // ---- backward W2 ----
            grad_w<<<OUTPUT, HIDDEN>>>(OUTPUT,HIDDEN,d_dW2,d_H,d_dW2,cur);
            // propagate to hidden
            /* (hidden back-prop kernel omitted here for brevity – same idea as grad_w +
               element-wise ReLU grad) */

            // ---- SGD update (W2 only, demo) ----
            saxpy<<< (OUTPUT*HIDDEN+255)/256,256 >>>(d_W2,d_dW2, -LR/cur, OUTPUT*HIDDEN);

            idx+=cur;
        }
        std::cout<<"epoch "<<epoch+1<<" done\n";
    }
    cudaEventRecord(stop);  cudaEventSynchronize(stop);
    float ms=0; cudaEventElapsedTime(&ms,start,stop);
    std::cout<<"GPU training time "<<ms/1000.0<<" s\n";
    return 0;
}
