#include <iostream>
#include <cstring>

#include "tokenizer.hpp"

// struct Tokenizer {
//     int                      max_token_length;
//     std::vector<std::string> vocab;
//     std::vector<float>       vocab_scores;
// };
// 
// Each token entry is of format {token_score, length in bytes, actual bytes of token}

Tokenizer createTokenizer(const char* filename, int vocab_size) {
    FILE* f = fopen(filename, "rb");
    if (f == nullptr) {
        std::perror(filename);
        exit(EXIT_FAILURE);
    }

    int maxTokenLen;
    if (fread(&maxTokenLen, sizeof(int), 1, f) != 1) {
        std::cerr << "failed to read maxTokenLen (max_token_length) inside createTokenizer\n";
    };

    std::vector<std::string> vocab  (vocab_size);
    std::vector<float>       scores (vocab_size);

    for (int i = 0; i < vocab_size; i++) {
        if (fread(scores.data() + i, sizeof(float), 1, f) != 1) {
            std::cerr << "failed to read score at i: " << i << " inside createTokenizer\n";
            fclose(f);
            exit(EXIT_FAILURE);
        }

        int tokenLen;
        if (fread(&tokenLen, sizeof(int), 1, f) != 1) {
            std::cerr << "failed to read tokenLen at i: " << i << " inside createTokenizer\n";
            fclose(f);
            exit(EXIT_FAILURE);
        }

        char* tok = new char[tokenLen];
        if (fread(tok, sizeof(char), tokenLen, f) != tokenLen) {
            std::cerr << "failed to read token at i: " << i << " inside createTokenizer\n";
            fclose(f);
            exit(EXIT_FAILURE);
        }
        vocab[i] = std::string(tok);
    }

    fclose(f);

    std::cout << "Read tokenizer.bin successfully.\n\n";

    return Tokenizer {
        .max_token_length = maxTokenLen,
        .vocab            = vocab,
        .vocab_scores     = scores,
    };
}

const char* decodeToken(int token_id, const Tokenizer& t) {
    return t.vocab[token_id].c_str();
}