#include <iostream>
#include <random>

#include "utils.hpp"

// mt19937: explicit seed, reproducible across platforms (unlike rand()/RAND_MAX,
// whose quality and period are implementation-defined). uniform_real_distribution
// produces values in the half-open range [0, 1), so this never returns exactly 1.0
// the way rand() / RAND_MAX could when rand() returned RAND_MAX.
static std::mt19937 rngEngine;
static std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

void seedRng(unsigned int seed) {
    rngEngine.seed(seed);
}

float getRandFloat() {
    return uniformDist(rngEngine);
}

// softmax algo. output will be written into the input array itself
void softmax(float* in, int n) {
    // softmax over 0..n
    // involves four passes over att
    // first pass
    float max_att = INT_MIN;
    for (int i = 0; i <= n; i++) {
        if (in[i] >= max_att) {
            max_att = in[i];
        }
    }

    // second pass. third pass sums up all the values. adding it here only
    float sum_att {0};
    for (int i = 0; i <= n; i++) {
        in[i] = expf(in[i] - max_att);
        sum_att += in[i];
    }

    // final pass
    for (int i = 0; i <= n; i++) {
        in[i] /= sum_att;
    }
}