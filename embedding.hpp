#ifndef _PXG_EMBEDDING_HPP_
#define _PXG_EMBEDDING_HPP_

#include <vector>
#include "config.hpp"

struct InputEmbedding {
    InputEmbedding() {
        for(int i = 0; i < VOCAB_SIZE; i++) {
            float* w = new float[HIDDEN_SIZE];
            embedding.push_back(w);
        }
    }
    ~InputEmbedding() {
        for(int i = 0; i < VOCAB_SIZE; i++) {
            float* w = embedding[i];
            delete w;
        }
    }
    void run(const int rank);
private:
    std::vector<float *> embedding;

};

#endif
