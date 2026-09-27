#include "sampler.hpp"

Sampler createSampler() {
    // these are default. TODO: make these configurable via user input
    return Sampler {
        .temperature    = 0,
        .topp           = 0.9,
    };
}

int sample(const Sampler& sm, std::vector<float>& logits) {
    if (sm.temperature == 0) {
        // skip sampling entirely
        return std::max_element(logits.begin(), logits.end()) - logits.begin();
    }

    return std::max_element(logits.begin(), logits.end()) - logits.begin();
    // TODO: remove return stmt above and finish this. i am sleepy right now
}