// Writes llama.cpp's logits at every position of a token sequence, for comparison with the
// PyTorch reference. Takes the usual llama.cpp options (-m, -ngl, -c, -b, -ub, ...) and:
//   TOKENS=<file>  int32 token ids (reference.py writes tokens.bin)
//   OUT=<file>     float32 logits, [n_tokens][n_vocab]
//   CHUNK=<n>      tokens per decode call (default: all at once, up to -b); 1 = one at a time
//   DUMP=<dir>     also write every F32 graph tensor whose name matches DUMP_RE (default
//                  "l_last-.*") to <dir>/<name>.<call>.bin, for layer-by-layer comparison
//
// Build against a llama.cpp build tree:
//   g++ -O2 -std=c++17 llama_logits.cpp -I$LLAMA/include -I$LLAMA/common -I$LLAMA/ggml/include \
//       -L$LLAMA/build/bin -lllama-common -lllama -lggml-base -Wl,-rpath,$LLAMA/build/bin -o llama-logits

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>

struct dump_state {
    std::string dir;
    std::regex  re;
    std::map<std::string, int> calls;
};

static bool dump_tensor(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = static_cast<dump_state *>(user_data);
    if (!std::regex_match(t->name, d->re) || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        return false;
    }
    if (ask) {
        return true;
    }
    std::vector<float> buf(ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    const int call = d->calls[t->name]++;
    std::ofstream f(d->dir + "/" + t->name + "." + std::to_string(call) + ".bin", std::ios::binary);
    f.write(reinterpret_cast<const char *>(buf.data()), buf.size() * sizeof(float));
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const char * tokens_path = std::getenv("TOKENS");
    const char * out_path    = std::getenv("OUT");
    if (tokens_path == nullptr || out_path == nullptr) {
        fprintf(stderr, "set TOKENS and OUT\n");
        return 1;
    }
    std::vector<llama_token> tokens;
    {
        std::ifstream f(tokens_path, std::ios::binary);
        int32_t t;
        while (f.read(reinterpret_cast<char *>(&t), sizeof(t))) {
            tokens.push_back(t);
        }
    }

    dump_state dump;
    if (std::getenv("DUMP")) {
        dump.dir = std::getenv("DUMP");
        dump.re  = std::regex(std::getenv("DUMP_RE") ? std::getenv("DUMP_RE") : "l_last-.*");
        params.cb_eval           = dump_tensor;
        params.cb_eval_user_data = &dump;
    }

    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "failed to load the model\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n       = (int) tokens.size();
    int chunk = std::getenv("CHUNK") ? std::atoi(std::getenv("CHUNK")) : n;
    if (chunk > (int) params.n_batch) {
        chunk = params.n_batch;
    }

    std::vector<float> logits((size_t) n * n_vocab);
    common_batch batch(ctx);
    for (int pos = 0; pos < n; pos += chunk) {
        const int m = std::min(chunk, n - pos);
        batch.clear();
        for (int j = 0; j < m; j++) {
            batch.add(tokens[pos + j], pos + j, 0, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "decode failed at position %d\n", pos);
            return 1;
        }
        for (int j = 0; j < m; j++) {
            const float * l = llama_get_logits_ith(ctx, j);
            std::copy(l, l + n_vocab, logits.begin() + (size_t) (pos + j) * n_vocab);
        }
    }
    std::ofstream out(out_path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(logits.data()), logits.size() * sizeof(float));
    fprintf(stderr, "wrote %d x %d logits to %s (chunk %d)\n", n, n_vocab, out_path, chunk);
    llama_backend_free();
    return 0;
}
