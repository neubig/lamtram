#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <lamtram/sentence.h>
#include <lamtram/dict-utils.h>

namespace dynet {
  class Dict;
}

namespace lamtram {

// A training target, where:
// * first is a dense vector of distributions
// * second is a sparse vector of distributions
// typedef std::pair<std::vector<float>, std::vector<std::pair<int, float> > > DistTarget;

// A base class implementing the functions necessary for calculation
class DistBase {

public:

  typedef std::vector<std::pair<int,float> > SparseData;
  typedef std::vector<std::pair<std::pair<int,int>,float> > BatchSparseData;

  DistBase(const std::string & sig) : ctxt_len_(0) { }
  virtual ~DistBase() { }

  // Get the signature of this class that uniquely identifies it for loading
  // at test time. In other words, the signature can collapse any information
  // only needed at training time.
  virtual std::string get_sig() const = 0;

  // Add stats from one sentence at training time for count-based models
  virtual void add_stats(const Sentence & sent) = 0;

  // Perform any final calculations on the stats
  virtual void finalize_stats() { }

  // Get the length of n-gram context that this model expects
  virtual size_t get_ctxt_len() const { return ctxt_len_; }

  // Get the number of contextual features we can expect from this model
  virtual size_t get_ctxt_size() const = 0;
  // Calculate the features
  virtual void calc_ctxt_feats(const Sentence & ctxt,
                               float* feats_out) const = 0;

  // Get the number of distributions we can expect from this model
  virtual size_t get_dense_size() const = 0;
  virtual size_t get_sparse_size() const = 0;
  // Calculate the probability of the last word in "ngram" given the context.
  // uniform_prob and unk_prob are probabilities of the uniform distribution,
  // and the unknown word penalty. DistTarget is the target distributions, and
  // the offset tell us the index to write to for dense or sparse distributions.
  virtual void calc_word_dists(const Sentence & ngram,
                               float uniform_prob,
                               float unk_prob,
                               std::vector<float> & trg_dense,
                               int & dense_offset,
                               SparseData & trg_sparse,
                               int & sparse_offset) const = 0;
  virtual void calc_all_word_dists(const Sentence & ctxt_ngram,
                                   int vocab_size,
                                   float uniform_prob,
                                   float unk_prob,
                                   std::vector<float> & trg_dense,
                                   int & dense_offset,
                                   BatchSparseData & trg_sparse,
                                   int & sparse_offset) const;

  // Read/write model. If dict is null, use numerical ids, otherwise strings.
  virtual void write(DictPtr dict, std::ostream & str) const = 0;
  virtual void read(DictPtr dict, std::istream & str) = 0;

protected:
  size_t ctxt_len_;  

};

typedef std::shared_ptr<DistBase> DistPtr;

}
