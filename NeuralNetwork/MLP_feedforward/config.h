#pragma once

#include <cstddef>

constexpr int IMG_W = 32;               // resize every sample to 32×32 grayscale
constexpr int IMG_H = 32;
constexpr int INPUT  = IMG_W * IMG_H;   // 1024
constexpr int HIDDEN = 512;             // one hidden layer
constexpr int OUTPUT = 62;              // 10 + 26 + 26
constexpr int BATCH  = 256;
constexpr int EPOCHS = 3;               // raise for a “real” run
constexpr float LR   = 0.01f;           // SGD step
