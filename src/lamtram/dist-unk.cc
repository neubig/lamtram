#include <lamtram/dist-unk.h>
#include <lamtram/macros.h>

using namespace lamtram;

DistUnk::DistUnk(const std::string & sig) : DistBase(sig) {
  if(sig != "unk")
    THROW_ERROR("Bad signature: " << sig);
}

void DistUnk::calc_word_dists(const Sentence & ngram,
                              float uniform_prob,
                              float unk_prob,
                              std::vector<float> & trg_dense,
                              int & dense_offset,
                              std::vector<std::pair<int,float> > & trg_sparse,
                              int & sparse_offset) const {
  trg_dense[dense_offset++] = (*ngram.rbegin() == 0 ? unk_prob : 0.f);
}
