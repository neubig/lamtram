#pragma once

#include <dynet/expr.h>
#include <lamtram/softmax-base.h>

namespace dynet { struct Parameter; }

namespace lamtram {

// An interface to a class that takes a vector as input
// (potentially batched) and calculates a probability distribution
// over words
class SoftmaxFull : public SoftmaxBase {

public:
  SoftmaxFull(const std::string & sig, int input_size, const DictPtr & vocab, dynet::Model & mod);
  ~SoftmaxFull() { };

  // Create a new graph
  virtual void NewGraph(dynet::ComputationGraph & cg) override;

  // Calculate training loss for one word
  virtual dynet::expr::Expression CalcLoss(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ngram, bool train) override;
  // Calculate training loss for multiple words
  virtual dynet::expr::Expression CalcLoss(dynet::expr::Expression & in, dynet::expr::Expression & prior, const std::vector<Sentence> & ngrams, bool train) override;
  
  // Calculate the full probability distribution
  virtual dynet::expr::Expression CalcProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ctxt, bool train) override;
  virtual dynet::expr::Expression CalcProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const std::vector<Sentence> & ctxt, bool train) override;
  virtual dynet::expr::Expression CalcLogProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ctxt, bool train) override;
  virtual dynet::expr::Expression CalcLogProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const std::vector<Sentence> & ctxt, bool train) override;

protected:
  dynet::Parameter p_sm_W_; // Softmax weights
  dynet::Parameter p_sm_b_; // Softmax bias

  dynet::expr::Expression i_sm_W_;
  dynet::expr::Expression i_sm_b_;

};

}
