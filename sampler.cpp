#include <iostream>
#include <vector>

#include "sampler.hpp"
#include "utils.hpp"

#define SEED_RNG            1
#define TOPP_DEFAULT        0.9
#define TEMPERATURE_DEFAULT 1.0

// struct ProbIndex {
//     float prob;
//     int index;
// };

// struct Sampler {
//     float temperature;
//     float topp;
//     ProbIndex* probIndex;       // buffer used for topp sampling
// };

// stores number of elements qualifying from topp, as well as their recurring sum
// internal data structure, hence not writing it in the hpp file
struct ToppResult {
    int n;
    float recSum;
};

Sampler createSampler(const int vocab_size) {
    // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    // these are default. TODO: make these configurable via user input

    std::cout << "Creating sampler with seed_rng: " << SEED_RNG << " and temperature: " << TEMPERATURE_DEFAULT << "\n\n";

    seedRng(SEED_RNG);

    return Sampler {
        .temperature    = TEMPERATURE_DEFAULT,
        .topp           = TOPP_DEFAULT,
        .probIndex      = new ProbIndex[vocab_size],
    };
}

ToppResult sample_topp(float* prob, float topp, ProbIndex* probIndex, int vocab_size) {
    // create probIndex
    int probIndexSize = 0;

    for (unsigned int i = 0; i < vocab_size; i++) {
        // creating a separate variable called probIndexSize because I know that I'll be changing this for loop later
        probIndex[i].prob = prob[i];
        probIndex[i].index = i;
        probIndexSize++;
    }

    std::sort(
        probIndex,
        probIndex + probIndexSize,
        [](ProbIndex& a, ProbIndex& b) {
            return a.prob > b.prob;
        }
    );

    float recSum = 0;
    int cut = vocab_size - 1;
    for (unsigned int i = 0; i < probIndexSize; i++) {
        recSum += probIndex[i].prob;
        if (recSum > topp) {
            cut = i;
            break;
        }
    }

    return ToppResult {
        .n      = cut + 1,
        .recSum = recSum,
    };
}

int sample(Sampler& sm, std::vector<float>& logits) {
    if (sm.temperature == 0) {
        // skip sampling entirely
        return std::max_element(logits.begin(), logits.end()) - logits.begin();
    }

    for (int i = 0; i < logits.size(); i++) {
        logits[i] /= sm.temperature;
    }

    softmax(logits.data(), logits.size());
    // logits holds probabilities post this

    ToppResult res = sample_topp(logits.data(), sm.topp, sm.probIndex, logits.size());
    int topp_size = res.n;

    float r = getRandFloat() * res.recSum;
    float recSum = 0;
    // cut index
    int cut = topp_size - 1;        // initialising as last index as fallback

    for (unsigned int i = 0; i < topp_size; i++) {
        recSum += sm.probIndex[i].prob;
        if (recSum > r) {
            cut = i;
            break;
        } 
    }

    return sm.probIndex[cut].index;
}