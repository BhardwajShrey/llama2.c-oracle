#include <iostream>
#include <cstdio>
#include <cstdlib>
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
    float* output;
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

void initWeights(const Config& config, Weights& w, void* data, bool sharedWeights) {
    // skip header. that's where floats begin
    float* p = reinterpret_cast<float*>(static_cast<char*>(data) + sizeof(Config));
    w.tok_embeddings    = p; p += (long long)config.vocab_size * config.dim;                    // tok_embeddings
    w.att_norm          = p; p += (long long)config.n_layers * config.dim;                      // att_norm
    w.wq                = p; p += (long long)config.n_layers * config.dim * config.dim;         // wq
    w.wk                = p; p += (long long)config.n_layers * config.dim * config.dim;         // wk 
    w.wv                = p; p += (long long)config.n_layers * config.dim * config.dim;         // wv
    w.wo                = p; p += (long long)config.n_layers * config.dim * config.dim;         // wo
    w.ffn_norm          = p; p += (long long)config.n_layers * config.dim;                      // ffn_norm
    w.w1                = p; p += (long long)config.n_layers * config.dim * config.hidden_dim;  // w1
    w.w2                = p; p += (long long)config.n_layers * config.dim * config.hidden_dim;  // w2
    w.w3                = p; p += (long long)config.n_layers * config.dim * config.hidden_dim;  // w3
    w.final_norm        = p; p += (long long)config.dim;                                        // final_norm

    if (sharedWeights) {
        w.output = w.tok_embeddings;
    } else {
        w.output = (p + (long long)config.seq_len * (config.dim / config.n_heads));
    }

    // long long advanced = p - reinterpret_cast<float*>(static_cast<char*>(data) + sizeof(Config));
    // std::cout << "advanced: " << advanced << " floats\n";
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

// x = input, g = w.att_norm (6 layers of 288 floats of weight), eps = 1e-5 (guards against divide by zero)
void rmsNorm(float* x, float* g, float eps, const int dim, std::vector<float>& out) {
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
void matmul(float* out, const float* x, const float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float acc = 0;
        size_t rowNum = i * n;

        for (int j = 0; j < n; j++) {
            acc += w[rowNum + j] * x[j];
        }
        out[i] = acc;
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

    std::vector<float> x(config.dim);
    getTokEmbedding(w, x.data(), 1, config.dim);

    if (dumpFloats("mine/embeddings.bin", x.data(), config.dim) == false) {
        std::cerr << "failed to dump data to mine/embeddings.bin";
    }

    std::vector<float> rmsOut(config.dim);
    // 1e-5 is not in Config -- it's hardcoded here to match run.c's rmsnorm()
    // and model.py's ModelArgs.norm_eps default (see GLOSSARY.md "Per-layer norms")
    rmsNorm(x.data(), w.att_norm, 1e-5, config.dim, rmsOut);

    if (dumpFloats("mine/att_norm.bin", rmsOut.data(), config.dim) == false) {
        std::cerr << "failed to dump data to mine/att_norm.bin";
    }

    std::vector<float> matmulOut(config.dim);
    matmul(matmulOut.data(), rmsOut.data(), w.wq, config.dim, config.dim);

    if (dumpFloats("mine/matmul_wq.bin", matmulOut.data(), config.dim) == false) {
        std::cerr << "failed to dump data to mine/matmul_wq.bin";
    }

    munmap(data, st.st_size);

    return 0;
}
