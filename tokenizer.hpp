#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>

struct Tokenizer {
    int                      max_token_length;
    std::vector<std::string> vocab;
    std::vector<float>       vocab_scores;
    unsigned char*           byte_pieces; // stores all single-byte strings
};

char* decodeToken(int token_id, const Tokenizer& t);

Tokenizer createTokenizer(const char* filename, int vocab_size);

#endif