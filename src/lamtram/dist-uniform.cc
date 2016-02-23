#include <lamtram/dist-uniform.h>
#include <lamtram/macros.h>

using namespace lamtram;

DistUniform::DistUniform(const std::string & sig) : DistBase(sig) {
  if(sig != "uniform")
    THROW_ERROR("Bad signature: " << sig);
}

void DistUniform::calc_word_dists(const Sentence & ngram,
                                  float uniform_prob,
                                  float unk_prob,
                                  std::vector<float> & trg_dense,
                                  int & dense_offset,
                                  std::vector<std::pair<int,float> > & trg_sparse,
                                  int & sparse_offset) const {
  trg_dense[dense_offset++] = (*ngram.rbegin() == 0 ? uniform_prob * unk_prob : uniform_prob);
}
