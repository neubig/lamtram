#pragma once

#include <lamtram/sentence.h>
#include <lamtram/dict-utils.h>
#include <memory>

namespace cnn { 
  class Model;
  class ComputationGraph;
  namespace expr { class Expression; }
}

namespace lamtram {

// An interface to a class that takes a vector as input
// (potentially batched) and calculates a probability distribution
// over words
class SoftmaxBase {

public:
  SoftmaxBase(const std::string & sig, int input_size, const DictPtr & vocab, cnn::Model & mod) : sig_(sig), input_size_(input_size), ctxt_len_(0), vocab_(vocab) { };
  ~SoftmaxBase() { };

  // Create a new graph
  virtual void NewGraph(cnn::ComputationGraph & cg) = 0;

  // Calculate training loss for one word. train_time indicates that we are training, in 
  // case we want to do something differently (such as dropout)
  virtual cnn::expr::Expression CalcLoss(cnn::expr::Expression & in, const Sentence & ngram, bool train) = 0;
  // Calculate training loss for a multi-word batch
  virtual cnn::expr::Expression CalcLoss(cnn::expr::Expression & in, const std::vector<Sentence> & ngrams, bool train) = 0;
  
  // Calculate the full probability distribution
  virtual cnn::expr::Expression CalcProbability(cnn::expr::Expression & in) = 0;
  virtual cnn::expr::Expression CalcLogProbability(cnn::expr::Expression & in) = 0;

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
