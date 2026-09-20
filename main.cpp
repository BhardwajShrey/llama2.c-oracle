#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

const char* filename = "out/stories15M.bin";

struct Config {
    int dim;         // 288  - the width of x
    int hidden_dim;  // 768  - FFN internal width
    int n_layers;    // 6
    int n_heads;     // 6
    int n_kv_heads;  // 6
    int vocab_size;  // 32000
    int seq_len;     // 256
};

struct Weights {
    float* tok_embeddings;
    float* att_norm;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* ffn_norm;
    float* w1;
    float* w2;
    float* w3;
    float* final_norm;
    float* cos_table;
    float* sin_table;
    float* output;
};

struct RunState {
    std::vector<float> x;               // dim      - the activation
    std::vector<float> xb;              // dim      - scratch after norm
    std::vector<float> xb2;             // dim      - scratch after matmul, post attention
    std::vector<float> q;               // dim
    std::vector<float> k;               // dim
    std::vector<float> v;               // dim
    std::vector<float> key_cache;       // n_layers * seq_len * dim
    std::vector<float> value_cache;     // n_layers * seq_len * dim
    std::vector<float> att;             // seq_len    - attention scores
    std::vector<float> hb;              // hidden_dim - FFN scratch
    std::vector<float> hb2;             // hidden_dim - FFN scratch
    std::vector<float> logits;          // vocab_size
};

void printFirstN(const char* msg, float* arr, int n = 5) {
    std::cout << msg << ": ";
    for (int i = 0; i < n; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n";
}

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
    std::cout << "floatCount processed: " << floatCount << "\n";
    return (floatCount * sizeof(float)) + sizeof(Config);
}

void readConfig(void* data, Config& config, bool& sharedWeights) {
    const Config* cfg_ptr = reinterpret_cast<const Config*>(data);
    config = *cfg_ptr;

    sharedWeights = config.vocab_size > 0;
    config.vocab_size = std::abs(config.vocab_size); 
}

void print_config(const Config& config, const bool& sharedWeights) {
    std::cout << "config.dim: "         << config.dim << "\n"
              << "config.hidden_dim: "  << config.hidden_dim << "\n"
              << "config.n_layers: "    << config.n_layers << "\n"
              << "config.n_heads: "     << config.n_heads << "\n"
              << "config.n_kv_heads: "  << config.n_kv_heads << "\n"
              << "config.vocab_size: "  << config.vocab_size << "\n"
              << "config.seq_len: "     << config.seq_len << "\n"
              << "shared weights: "     << sharedWeights << "\n";  
}

void initWeights(const Config& config, Weights& w, void* data, bool sharedWeights) {
    // skip header. that's where floats begin
    float* p = reinterpret_cast<float*>(static_cast<char*>(data) + sizeof(Config));
    w.tok_embeddings    = p; p += (long long)config.vocab_size * config.dim;                                    // tok_embeddings
    w.att_norm          = p; p += (long long)config.n_layers * config.dim;                                      // att_norm
    // TODO: wk/wv offsets assume n_kv_heads == n_heads (kv_dim == dim). run.c uses
    // kv_dim = (dim * n_kv_heads) / n_heads for these two -- fix once attention block is done.
    w.wq                = p; p += (long long)config.n_layers * config.dim * config.dim;                         // wq
    w.wk                = p; p += (long long)config.n_layers * config.dim * config.dim;                         // wk
    w.wv                = p; p += (long long)config.n_layers * config.dim * config.dim;                         // wv
    w.wo                = p; p += (long long)config.n_layers * config.dim * config.dim;                         // wo
    w.ffn_norm          = p; p += (long long)config.n_layers * config.dim;                                      // ffn_norm
    w.w1                = p; p += (long long)config.n_layers * config.dim * config.hidden_dim;                  // w1
    w.w2                = p; p += (long long)config.n_layers * config.hidden_dim * config.dim;                  // w2
    w.w3                = p; p += (long long)config.n_layers * config.dim * config.hidden_dim;                  // w3
    w.final_norm        = p; p += (long long)config.dim;                                                        // final_norm
    w.cos_table         = p; p += (long long)(long long)config.seq_len * (config.dim / config.n_heads / 2);     // cosine tables
    w.sin_table         = p; p += (long long)(long long)config.seq_len * (config.dim / config.n_heads / 2);     // sine tables

    if (sharedWeights) {
        w.output = w.tok_embeddings;
    } else {
        w.output = p;
    }

    // long long advanced = p - reinterpret_cast<float*>(static_cast<char*>(data) + sizeof(Config));
    // std::cout << "advanced: " << advanced << " floats\n";
}

RunState createRunState(const Config& config) {
    int dim         {config.dim},
        n_layers    {config.n_layers},
        seq_len     {config.seq_len},
        hidden_dim  {config.hidden_dim},
        vocab_size  {config.vocab_size};

    return RunState {
        .x              = std::vector<float> (dim),                               // x  - dim      - the activation
        .xb             = std::vector<float> (dim),                               // xb - dim      - scratch after norm (RMS)
        .xb2            = std::vector<float> (dim),                               // xb2 - dim      - scratch after matmul with wo post attention block
        .q              = std::vector<float> (dim),                               // q  - dim
        .k              = std::vector<float> (dim),                               // k  - dim
        .v              = std::vector<float> (dim),                               // v - dim
        // TODO: sized with dim, should be kv_dim = (dim * n_kv_heads) / n_heads -- see cacheOffset
        .key_cache      = std::vector<float> (n_layers * seq_len * dim),          // n_layers * seq_len * dim
        .value_cache    = std::vector<float> (n_layers * seq_len * dim),          // n_layers * seq_len * dim
        .att            = std::vector<float> (seq_len),                           // seq_len
        .hb             = std::vector<float> (hidden_dim),                        // hidden_dim
        .hb2            = std::vector<float> (hidden_dim),                        // hidden_dim
        .logits         = std::vector<float> (vocab_size),                        // vocab_size
    };
}

void getTokEmbedding(const Weights& w, float* x, int tokId, int dim) {
    // TODO: Add a check for whether the token id is out of range. Decide what behaviour must come
    long long xSize = static_cast<long long>(dim);
    float* startP = w.tok_embeddings + (tokId * xSize);
    std::copy(startP, startP + xSize, x);
}

bool dumpFloats(const char* path, const float* arr, long long count) {
    if (count <= 0) {
        std::cerr << "count is equal to or less than zero.\n";
        return false;
    }

    std::FILE* file = std::fopen(path, "wb");
    if (file == nullptr) {
        std::perror(path);
        return false;
    }

    size_t elemsWritten = std::fwrite(arr, sizeof(float), static_cast<size_t>(count), file);

    if (elemsWritten != count) {
        std::cerr << "Expected elemsWritten: " << count << ". Actual: " << elemsWritten << "\n";
        std::fclose(file);
        return false;
    }

    std::fclose(file);

    return true;
}

// x = input, g = a per-layer norm weight (att_norm or ffn_norm, caller offsets by layer),
// dim floats each. eps guards against divide by zero and is hardcoded below (not a
// parameter) -- it's not in Config either; matches run.c's rmsnorm() and model.py's
// ModelArgs.norm_eps default (see GLOSSARY.md "Per-layer norms")
void rmsNorm(float* x, float* g, const int dim, float* out) {
    const float eps {1e-5f};

    float sumSquares {0};

    for (int i = 0; i < dim; i++) {
        sumSquares += x[i] * x[i];
    }

    float mean {sumSquares / static_cast<float>(dim)};
    mean = mean + eps;
    
    float scale = {1.0f / sqrtf(mean)};     // reciprocal square root

    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] * scale * g[i]);
    }
}

// this is a d * n matrix by n * 1 matrix multiplication. End result is d * 1
// DISCLAIMER: if the same buffer is passed as both out and x, matmul will be corrupted
// d = number of weights in w, also equal to out's dimension
// n = dimension of each weight in w, also equal to dimension of x
void matmul(float* out, const float* x, const float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float acc = 0;
        size_t rowNum = i * static_cast<size_t>(n);

        for (int j = 0; j < n; j++) {
            acc += w[rowNum + j] * x[j];
        }
        out[i] = acc;
    }
}

void rope(float* cosines, float* sines, const Config& config, int pos, float* q) {
    // imagine cosines and sines as tables of seq_len rows and (head_dim / 2) entries per row
    // at position 0 angles are zero. cosines will all be 1 and sines will be 0

    int head_dim = (config.dim / config.n_heads);

    cosines += (pos * head_dim / 2);
    sines += (pos * head_dim / 2);

    // printFirstN("cosines", cosines, 24);
    // printFirstN("sines", sines, 24);

    // break token down into n_heads (6) chunks of 48 floats each. Process 2 at a time

    for (int h = 0; h < config.n_heads; h++) {
        for (int i = 0; i < head_dim / 2; i++) {
            int base = (h * head_dim) + (2 * i);
            float x0 = q[base];
            float x1 = q[base + 1];

            q[base]     = x0 * cosines[i] - x1 * sines[i];
            q[base + 1] = x0 * sines[i] + x1 * cosines[i];
        }
    }
}

// k & v caches can be imagined as a 3-d array of dimensions n_layers * seq_len * dim
// n_layer arrays of (seq_len arrays of length dim)
// TODO: dim here should be kv_dim = (dim * n_kv_heads) / n_heads, same as RunState's
// key_cache/value_cache sizing in createRunState -- fix once attention block is done.
int cacheOffset(int l, int pos, const Config& config) {
    return l * (config.seq_len * config.dim) + (pos * config.dim);
}

void writeToCache(const float* in, float* cache, int layer, int pos, const Config& config) {
    std::copy(in, in + config.dim, cache + cacheOffset(layer, pos, config));
}

float dot(const float* a, const float* b, int len) {
    float dot_product {0};

    for (int i = 0; i < len; i++) {
        dot_product += (a[i] * b[i]);
    }

    return dot_product;
}

float sigmoid(float v) {
    return 1.f / (1.f + expf(-v));
}

void swiGLU(float* out, float* hb, float* hb2, int len) {
    for (int i = 0; i < len; i++) {
        out[i] = hb[i] * sigmoid(hb[i]) * hb2[i];
    }
}

void forward(RunState& s, const Config& config, const Weights& w, int token_id, int pos) {
    getTokEmbedding(w, s.x.data(), token_id, config.dim);

    if (dumpFloats("mine/embeddings.bin", s.x.data(), config.dim) == false) {
        std::cerr << "failed to dump data to mine/embeddings.bin";
    }

    for (int layer = 0; layer < config.n_layers; layer++) {
        std::string layerSuffix {"_layer_" + std::to_string(layer) + ".bin"};

        rmsNorm(s.x.data(), w.att_norm + (layer * config.dim), config.dim, s.xb.data());
        std::string attNormPath {"mine/att_norm" + layerSuffix};
        if (dumpFloats(attNormPath.c_str(), s.xb.data(), config.dim) == false) {
            std::cerr << "failed to dump data to " << attNormPath;
        }

        long long kqvOffset {layer * config.dim * config.dim};

        matmul(s.q.data(), s.xb.data(), w.wq + kqvOffset, config.dim, config.dim);
        std::string matmulWqPath {"mine/matmul_wq" + layerSuffix};
        if (dumpFloats(matmulWqPath.c_str(), s.q.data(), config.dim) == false) {
            std::cerr << "failed to dump data to " << matmulWqPath;
        }

        // TODO: n/d here should be kv_dim = (dim * n_kv_heads) / n_heads, not dim -- same
        // GQA assumption as the wk/wv offsets in initWeights. Fix once attention block is done.
        matmul(s.k.data(), s.xb.data(), w.wk + kqvOffset, config.dim, config.dim);
        std::string matmulWkPath {"mine/matmul_wk" + layerSuffix};
        if (dumpFloats(matmulWkPath.c_str(), s.k.data(), config.dim) == false) {
            std::cerr << "failed to dump data to " << matmulWkPath;
        }

        matmul(s.v.data(), s.xb.data(), w.wv + kqvOffset, config.dim, config.dim);
        std::string matmulWvPath {"mine/matmul_wv" + layerSuffix};
        if (dumpFloats(matmulWvPath.c_str(), s.v.data(), config.dim) == false) {
            std::cerr << "failed to dump data to " << matmulWvPath;
        }

        // ROPE for q and k, not v
        rope(w.cos_table, w.sin_table, config, pos, s.q.data());
        rope(w.cos_table, w.sin_table, config, pos, s.k.data());

        // attention stage begins

        // copy v and post rope k for layer 0 and pos 0 into caches
        writeToCache(s.k.data(), s.key_cache.data(), layer, pos, config);
        writeToCache(s.v.data(), s.value_cache.data(), layer, pos, config);

        int head_dim {config.dim / config.n_heads};

        for (int h = 0; h < config.n_heads; h++) {
            // processing happens per head. each head fully computes its own scores before the next head starts. softmax and weighted sum will reside in this for loop only
            int slice = h * head_dim;       // what index does the head begin at. slice = one head of len head_dim

            // calculating score
            for (int t = 0; t <= pos; t++) {
                float* k_t = s.key_cache.data() + cacheOffset(layer, t, config) + slice;
                float score = dot(s.q.data() + slice, k_t, head_dim) / sqrtf(head_dim);
                s.att[t] = score;
            }

            // softmax over 0..pos
            // involves four passes over att
            // first pass
            float max_att = INT_MIN;
            for (int i = 0; i <= pos; i++) {
                if (s.att[i] >= max_att) {
                    max_att = s.att[i];
                }
            }

            // second pass. third pass sums up all the values. adding it here only
            float sum_att {0};
            for (int i = 0; i <= pos; i++) {
                s.att[i] = expf(s.att[i] - max_att);
                sum_att += s.att[i];
            }

            // final pass
            for (int i = 0; i <= pos; i++) {
                s.att[i] /= sum_att;
            }

            // weighted sum
            // zero the output slice (re-use xb used for rms earlier)
            for (int i = 0; i < head_dim; i++) {
                s.xb[slice + i] = 0;
            }

            for (int t = 0; t <= pos; t++) {
                float* v_t = s.value_cache.data() + cacheOffset(layer, t, config) + slice;

                for (int d = 0; d < head_dim; d++) {
                    s.xb[slice + d] += (s.att[t] * v_t[d]);
                }
            }
        }

        std::string attXbPath {"mine/att_xb" + layerSuffix};
        if (dumpFloats(attXbPath.c_str(), s.xb.data(), config.dim) == false) {
            std::cerr << "failed to dump data from s.xb to " << attXbPath;
        }

        matmul(s.xb2.data(), s.xb.data(), w.wo + kqvOffset, config.dim, config.dim);

        std::string attXb2Path {"mine/att_xb2" + layerSuffix};
        if (dumpFloats(attXb2Path.c_str(), s.xb2.data(), config.dim) == false) {
            std::cerr << "failed to dump data from s.xb2 to " << attXb2Path;
        }

        for (int i = 0; i < config.dim; i++) {
            s.x[i] += s.xb2[i];
        }

        // Attention ends

        // FEED FORWARD BEGINS
        // w1 and w3 are dim * hidden_dim matrices. w2 is hidden_dim * dim
        long long w1w3offset {layer * config.dim * config.hidden_dim};
        long long w2offset   {layer * config.hidden_dim * config.dim};      // yeah yeah its the same as w1w3offset, this is more about the principle

        rmsNorm(s.x.data(), w.ffn_norm + (layer * config.dim), config.dim, s.xb.data());
        std::string ffnNormPath {"mine/ffn_norm" + layerSuffix};
        if (dumpFloats(ffnNormPath.c_str(), s.xb.data(), config.dim) == false) {
            std::cerr << "failed to dump data from s.xb to " << ffnNormPath;
        }

        matmul(s.hb.data(), s.xb.data(), w.w1 + w1w3offset, config.dim, config.hidden_dim);          // dim -> hidden_dim
        std::string ffnW1Path {"mine/ffn_w1" + layerSuffix};
        if (dumpFloats(ffnW1Path.c_str(), s.hb.data(), config.hidden_dim) == false) {
            std::cerr << "failed to dump data from s.hb to " << ffnW1Path;
        }

        matmul(s.hb2.data(), s.xb.data(), w.w3 + w1w3offset, config.dim, config.hidden_dim);         // dim -> hidden_dim
        std::string ffnW3Path {"mine/ffn_w3" + layerSuffix};
        if (dumpFloats(ffnW3Path.c_str(), s.hb2.data(), config.hidden_dim) == false) {
            std::cerr << "failed to dump data from s.hb2 to " << ffnW3Path;
        }

        swiGLU(s.hb.data(), s.hb.data(), s.hb2.data(), config.hidden_dim);                           // elementwise, 768 floats

        std::vector<float> out(config.dim);
        matmul(out.data(), s.hb.data(), w.w2 + w2offset, config.hidden_dim, config.dim);
        std::string ffnW2Path {"mine/ffn_w2" + layerSuffix};
        if (dumpFloats(ffnW2Path.c_str(), out.data(), config.dim) == false) {
            std::cerr << "failed to dump data from out to " << ffnW2Path;
        }

        for (int i = 0; i < config.dim; i++) {
            s.x[i] += out[i];
        }
    }
}

int main() {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        std::perror(filename);
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        std::cerr << "Error running fstat on: " << filename << ", with fd: " << fd << "\n";
        return 1;
    }

    void* data = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) {
        std::cerr << "mmap on filename: " << filename << ", failed.\n";
        return 1;
    }

    Config config {};
    bool sharedWeights {};

    readConfig(data, config, sharedWeights);
    
    print_config(config, sharedWeights); 

    long long fileSizeExpected = expected_file_size(config);
    long long fileSizeActual = st.st_size;

    std::cout << "expected: " << fileSizeExpected << "\n";
    std::cout << "actual:   " << fileSizeActual << "\n";
    std::cout << "gap:      " << fileSizeActual - fileSizeExpected << " bytes = "
            << (fileSizeActual - fileSizeExpected) / 4 << " floats\n";

    Weights w {};

    initWeights(config, w, data, sharedWeights);

    // printFirstN("tok_embeddings", w.tok_embeddings);
    // printFirstN("wq", w.wq);

    RunState s = createRunState(config);

    // some sane defaults. For testing
    int token_id {1}, pos {0};

    forward(s, config, w, token_id, pos);

    munmap(data, st.st_size);

    return 0;
}
