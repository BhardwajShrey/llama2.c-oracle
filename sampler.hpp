#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include <vector>

struct Sampler {
    float temperature;
    float topp;
};

Sampler createSampler();

int sample(const Sampler& sm, std::vector<float>& logits);

#endif