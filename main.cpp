#include <iostream>
#include <cstdio>
#include <vector>
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

int main() {
    std::cout << "Opening file... \n";

    const char* filename = "out/stories15M.bin";
    FILE* file = std::fopen(filename, "rb");

    if (file == nullptr) {
        std::perror(filename);
        return 1;
    }

    Config config {};

    size_t blocksRead = std::fread(&config, sizeof(Config), 1, file);

    if (blocksRead != 1) {
        std::cout << "fread for config header failed. Did not return header size of 7\n";
        std::fclose(file);
        return 1;
    }

    bool sharedWeights = config.vocab_size > 0;
    config.vocab_size = std::abs(config.vocab_size);

    std::cout << "config.dim: "         << config.dim << "\n"
              << "config.hidden_dim: "  << config.hidden_dim << "\n"
              << "config.n_layers: "    << config.n_layers << "\n"
              << "config.n_heads: "     << config.n_heads << "\n"
              << "config.n_kv_heads: "  << config.n_kv_heads << "\n"
              << "config.vocab_size: "  << config.vocab_size << "\n"
              << "config.seq_len: "     << config.seq_len << "\n"
              << "shared weights: "     << sharedWeights;
    
    std::cout << "\nClosing file. \n";
    std::fclose(file);
    
    return 0;
}