#pragma once

#include <lamtram/sentence.h>
#include <lamtram/dict-utils.h>
#include <dynet/expr.h>
#include <memory>

namespace dynet { 
  class Model;
  struct ComputationGraph;
}

namespace lamtram {

// An interface to a class that takes a vector as input
// (potentially batched) and calculates a probability distribution
// over words
class SoftmaxBase {

public:
  SoftmaxBase(const std::string & sig, int input_size, const DictPtr & vocab, dynet::ParameterCollection & mod) : sig_(sig), input_size_(input_size), ctxt_len_(0), vocab_(vocab) { };
  ~SoftmaxBase() { };

  // Create a new graph
  virtual void NewGraph(dynet::ComputationGraph & cg) = 0;

  // Calculate training loss for one word. train_time indicates that we are training, in 
  // case we want to do something differently (such as dropout)
  virtual dynet::Expression CalcLoss(dynet::Expression & in, dynet::Expression & prior, const Sentence & ngram, bool train) = 0;
  // Calculate training loss for a multi-word batch
  virtual dynet::Expression CalcLoss(dynet::Expression & in, dynet::Expression & prior, const std::vector<Sentence> & ngrams, bool train) = 0;

  // Calculate loss using cached info
  virtual dynet::Expression CalcLossCache(dynet::Expression & in, dynet::Expression & prior, int cache_id, const Sentence & ngram, bool train) {
    return CalcLoss(in, prior, ngram, train);
  }
  virtual dynet::Expression CalcLossCache(dynet::Expression & in, dynet::Expression & prior, const std::vector<int> & cache_ids, const std::vector<Sentence> & ngrams, bool train) {
    return CalcLoss(in, prior, ngrams, train);
  }
  
  // Calculate the full probability distribution
  virtual dynet::Expression CalcProb(dynet::Expression & in, dynet::Expression & prior, const Sentence & ctxt, bool train) = 0;
  virtual dynet::Expression CalcProb(dynet::Expression & in, dynet::Expression & prior, const std::vector<Sentence> & ctxt, bool train) = 0;
  virtual dynet::Expression CalcLogProb(dynet::Expression & in, dynet::Expression & prior, const Sentence & ctxt, bool train) = 0;
  virtual dynet::Expression CalcLogProb(dynet::Expression & in, dynet::Expression & prior, const std::vector<Sentence> & ctxt, bool train) = 0;

  virtual dynet::Expression CalcProbCache(dynet::Expression & in, dynet::Expression & prior, int cache_id,                       const Sentence & ctxt, bool train) { return CalcProb(in,prior,ctxt,train); }
  virtual dynet::Expression CalcProbCache(dynet::Expression & in, dynet::Expression & prior, const Sentence & cache_ids, const std::vector<Sentence> & ctxt, bool train) { return CalcProb(in,prior,ctxt,train); }
  virtual dynet::Expression CalcLogProbCache(dynet::Expression & in, dynet::Expression & prior, int cache_id,                       const Sentence & ctxt, bool train) { return CalcLogProb(in,prior,ctxt,train); }
  virtual dynet::Expression CalcLogProbCache(dynet::Expression & in, dynet::Expression & prior, const Sentence & cache_ids, const std::vector<Sentence> & ctxt, bool train) { return CalcLogProb(in,prior,ctxt,train); }

  // Cache data for the entire training corpus if necessary
  //  data is the data, set_ids is which data set the sentences belong to
  virtual void Cache(const std::vector<Sentence> & sents, const std::vector<int> & set_ids, std::vector<Sentence> & cache_ids) { }

  // Update the fold by loading necessary data, etc.
  virtual void UpdateFold(int fold_id) { }

  virtual const std::string & GetSig() const { return sig_; }
  virtual int GetInputSize() const { return input_size_; }
  virtual int GetCtxtLen() const { return ctxt_len_; }

protected:
  std::string sig_;
  int input_size_;
  int ctxt_len_;
  DictPtr vocab_;

};

typedef std::shared_ptr<SoftmaxBase> SoftmaxPtr;

}
