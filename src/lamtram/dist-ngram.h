#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <lamtram/sentence.h>
#include <lamtram/dist-base.h>
#include <lamtram/hashes.h>

namespace lamtram {

struct DistNgramCounts {
  DistNgramCounts() : first(0), second(0), third(0) { }
  DistNgramCounts(int f, int s, int t) : first(f), second(s), third(t) { }
  int first;  // the actual count of the ngram
  int second; // the count of the context
  int third;  // the number of unique words following the context
};

// A class for the n-gram distribution
class DistNgram : public DistBase {

public:

  // Signature should be of the form
  // 1) ngram
  // 2) comb/split/all: where "comb" means combine together counts according to
  //    a heuristic, split means calculate each distribution separately, and all
  //    means to use both
  // 3) lin/wb/mkn: where "lin" means linear, "wb" means witten bell, and "mkn"
  //    means modified kneser ney
  DistNgram(const std::string & sig);
  virtual ~DistNgram() { }

  // Get the signature of this class that uniquely identifies it for loading
  // at test time. In other words, the signature can collapse any information
  // only needed at training time.
  virtual std::string get_sig() const override;

  // Add stats from one sentence at training time for count-based models
  virtual void add_stats(const Sentence & sent) override;

  // Perform finalization on stats
  virtual void finalize_stats() override;

  // Get the number of ctxtual features we can expect from this model
  virtual size_t get_ctxt_size() const override;
  // And calculate these features
  virtual void calc_ctxt_feats(const Sentence & ctxt, float * feats_out) const override;

  // Get the number of distributions we can expect from this model
  virtual size_t get_dense_size() const override { return heuristics_ ? 1 : ngram_len_; }
  virtual size_t get_sparse_size() const override { return 0; }
  virtual void calc_word_dists(const Sentence & ngram,
                               float uniform_prob,
                               float unk_prob,
                               std::vector<float> & trg_dense,
                               int & dense_offset,
                               SparseData & trg_sparse,
                               int & sparse_offset) const override;
  virtual void calc_all_word_dists(const Sentence & ctxt_ngram,
                                   int vocab_size,
                                   float uniform_prob,
                                   float unk_prob,
                                   std::vector<float> & trg_dense,
                                   int & dense_offset,
                                   BatchSparseData & trg_sparse,
                                   int & sparse_offset) const override;

  // Read/write model. If dict is null, use numerical ids, otherwise strings.
  virtual void write(DictPtr dict, std::ostream & str) const override;
  virtual void read(DictPtr dict, std::istream & str) override;

  // Create the context
  int get_ctxt_id(const Sentence & ngram);
  int get_tmp_ctxt_id(const Sentence & ngram);
  int get_existing_ctxt_id(const Sentence & ngram) const;
  

protected:

  typedef enum { SMOOTH_LIN, SMOOTH_MABS, SMOOTH_MKN } SmoothType;

  // Assume we have 3-grams
  // ** Standard
  //  - After add_stats
  //   - 3-gram counts will be held in mapping_[ngram]
  //   - 2-gram and 1-gram counts will be held in ctxt_cnts_[mapping_[ngram]].word_cnt
  //  - After finalize stats
  //   - All context counts will be held in ctxt_cnts_[mapping_[ngram]].ctxt_true
  // ** KN
  //  - After add_stats
  //   - 3-gram counts added to mapping_[ngram], and 2-gram to aux_cnts_[mapping_[ngram]]
  //  - After finalize stats (for abc)
  //   - If 3-gram
  //    - The full count to ctxt_cnts_[ab].ctxt_true
  //   - One to ctxt_cnts_[b].ctxt_true and ctxt_cnts_[bc]

  // A mapping from either counts or positions in the count array, depending
  // on whether the n-gram is the longest allowed.
  std::unordered_map<Sentence, int> mapping_, tmp_mapping_;
  // Counts for word context, etc
  std::vector<DistNgramCounts> ctxt_cnts_;
  // Discounted counts
  std::vector<float> disc_ctxt_cnts_;
  // Discounts
  std::vector<std::vector<float> > discounts_;
  // Auxilliary counts only used when calculating KN
  std::vector<int> aux_cnts_;
  // The position of the contexts, smoothing type, and number of distributions
  std::vector<int> ctxt_pos_;
  SmoothType smoothing_;
  size_t ngram_len_;
  bool heuristics_;

};

}
