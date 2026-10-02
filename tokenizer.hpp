#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <unordered_map>
#include <vector>

struct Tokenizer {
    int                                       max_token_length;
    std::vector<std::string>                  vocab;                 // tokenId -> token lookup
    std::vector<float>                        vocab_scores;
    unsigned char*                            byte_pieces;           // stores all single-byte strings
    std::unordered_map<std::string, int>      tokens;                // tokens -> tokenId lookup. reverse of vocab
};

char* decodeToken(int token_id, int prev, const Tokenizer& t);

Tokenizer createTokenizer(const char* filename, int vocab_size);

#endif