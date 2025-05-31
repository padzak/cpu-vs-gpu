#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

using Matrix = std::vector<std::vector<float>>;

// Fill a matrix with random floats
void random_init(Matrix& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& row : mat)
        for (auto& val : row)
            val = dist(gen);
}

// Matrix multiplication
Matrix matmul(const Matrix& A, const Matrix& B) {
    size_t m = A.size(), n = B[0].size(), k = B.size();
    Matrix C(m, std::vector<float>(n, 0.0f));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t l = 0; l < k; ++l)
                C[i][j] += A[i][l] * B[l][j];
    return C;
}

// ReLU activation
void relu(Matrix& mat) {
    for (auto& row : mat)
        for (auto& val : row)
            val = std::max(0.0f, val);
}

int main() {
    const int batch = 1024, input = 1000, h1 = 2048, h2 = 1024, output = 10;

    // Create layers
    Matrix X(batch, std::vector<float>(input));
    Matrix W1(input, std::vector<float>(h1));
    Matrix W2(h1, std::vector<float>(h2));
    Matrix W3(h2, std::vector<float>(output));

    random_init(X);
    random_init(W1);
    random_init(W2);
    random_init(W3);

    auto start = std::chrono::high_resolution_clock::now();

    // Forward pass
    Matrix Z1 = matmul(X, W1);
    relu(Z1);
    Matrix Z2 = matmul(Z1, W2);
    relu(Z2);
    Matrix Z3 = matmul(Z2, W3);  // Output layer (no activation)

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "CPU forward pass time: " << elapsed << " seconds\n";
    return 0;
}
