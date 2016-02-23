#include <lamtram/dist-base.h>
#include <lamtram/macros.h>
#include <cassert>

using namespace lamtram;
using namespace std;


void DistBase::calc_all_word_dists(const Sentence & ctxt_ngram,
                                   int vocab_size,
                                   float uniform_prob,
                                   float unk_prob,
                                   std::vector<float> & trg_dense,
                                   int & dense_offset,
                                   DistBase::BatchSparseData & trg_sparse,
                                   int & sparse_offset) const {
  // cerr << "calc_all_word_dists: " << ctxt_ngram << endl;
  assert(trg_dense.size() % vocab_size == 0);
  int dense_size = trg_dense.size()/vocab_size;
  Sentence ngram(ctxt_ngram); ngram.push_back(0);
  for(int i = 0; i < vocab_size; i++) {
    *ngram.rbegin() = i;
    int my_dense_offset = dense_offset + i*dense_size, my_sparse_offset = sparse_offset;
    SparseData my_trg_sparse;
    calc_word_dists(ngram, uniform_prob, unk_prob, trg_dense, my_dense_offset, my_trg_sparse, my_sparse_offset);
    for(auto & val : my_trg_sparse)
      trg_sparse.push_back(make_pair(make_pair(i, val.first), val.second));
  }
  sparse_offset += get_sparse_size();
  dense_offset += get_dense_size();
}
