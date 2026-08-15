#include <iostream>
#include <cstdio>
#include <cstdlib>

struct Config {
    int dim;         // 288  - the width of x
    int hidden_dim;  // 768  - FFN internal width
    int n_layers;    // 6
    int n_heads;     // 6
    int n_kv_heads;  // 6
    int vocab_size;  // 32000
    int seq_len;     // 256
};

long long expected_file_size(const Config& config) {
    long long floatCount = 0;

    // TODO: use static_cast<long long>() instead of (long long) --- C++ style vs C style cast
    floatCount += (long long)config.vocab_size * config.dim;                    // tok_embeddings
    floatCount += (long long)config.n_layers * config.dim;                      // att_norm
    floatCount += (long long)config.n_layers * config.dim * config.dim;         // wq
    floatCount += (long long)config.n_layers * config.dim * config.dim;         // wk 
    floatCount += (long long)config.n_layers * config.dim * config.dim;         // wv
    floatCount += (long long)config.n_layers * config.dim * config.dim;         // wo
    floatCount += (long long)config.n_layers * config.dim;                      // ffn_norm
    floatCount += (long long)config.n_layers * config.dim * config.hidden_dim;  // w1
    floatCount += (long long)config.n_layers * config.dim * config.hidden_dim;  // w2
    floatCount += (long long)config.n_layers * config.dim * config.hidden_dim;  // w3
    floatCount += (long long)config.dim;                                        // final_norm

    // computed size of weights plus size of header
    return (floatCount * sizeof(float)) + sizeof(Config);
}

bool readConfig(FILE* file, Config& config, bool& sharedWeights) {
    size_t blocksRead = std::fread(&config, sizeof(Config), 1, file);

    if (blocksRead != 1) {
        std::cerr << "fread for config header failed. blocksRead returned: " << blocksRead << "\n";
        return false;
    }

    sharedWeights = config.vocab_size > 0;
    config.vocab_size = std::abs(config.vocab_size); 
              
    return true;
}

void print_config(const Config& config, bool& sharedWeights) {
    std::cout << "config.dim: "         << config.dim << "\n"
              << "config.hidden_dim: "  << config.hidden_dim << "\n"
              << "config.n_layers: "    << config.n_layers << "\n"
              << "config.n_heads: "     << config.n_heads << "\n"
              << "config.n_kv_heads: "  << config.n_kv_heads << "\n"
              << "config.vocab_size: "  << config.vocab_size << "\n"
              << "config.seq_len: "     << config.seq_len << "\n"
              << "shared weights: "     << sharedWeights << "\n";  
}

int main() {
    std::cout << "Opening file... \n";

    const char* filename = "out/stories15M.bin";
    FILE* file = std::fopen(filename, "rb");

    if (file == nullptr) {
        std::perror(filename);
        return 1;
    }

    Config config {};
    bool sharedWeights {};

    if (!readConfig(file, config, sharedWeights)) {
        std::cerr << "Unable to read config. Terminating...\n";
        std::fclose(file);
        return 1;
    }
    
    print_config(config, sharedWeights); 

    long long fileSizeExpected = expected_file_size(config);

    std::fseek(file, 0, SEEK_END);
    long long fileSizeActual = std::ftell(file);
    std::rewind(file);

    std::cout << "expected: " << fileSizeExpected << "\n";
    std::cout << "actual:   " << fileSizeActual << "\n";
    std::cout << "gap:      " << fileSizeActual - fileSizeExpected << " bytes = "
            << (fileSizeActual - fileSizeExpected) / 4 << " floats\n";

    
    std::cout << "\nClosing file. \n";
    std::fclose(file);
    
    return 0;
}
