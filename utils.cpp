#include <iostream>

#include "utils.hpp"

float getRandFloat() {
    float rn = static_cast<float>(rand()) / RAND_MAX;
    return rn;
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