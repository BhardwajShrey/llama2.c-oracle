#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include <vector>

struct ProbIndex {
    float prob;
    int index;
};

struct Sampler {
    float temperature;
    float topp;
    ProbIndex* probIndex;       // buffer used for topp sampling
};

Sampler createSampler(const int vocab_size);

int sample(Sampler& sm, std::vector<float>& logits);

#endif