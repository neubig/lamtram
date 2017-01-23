#pragma once

#include <dynet/expr.h>
#include <lamtram/softmax-base.h>
#include <lamtram/softmax-factory.h>
#include <boost/algorithm/string.hpp>

namespace dynet { struct Parameter; }

namespace lamtram {

// An interface to a class that takes a vector as input
// (potentially batched) and calculates a probability distribution
// over words
class SoftmaxMultiLayer : public SoftmaxBase {
    friend class CnpyUtils;

public:
  SoftmaxMultiLayer(const std::string & sig, int input_size, const DictPtr & vocab, dynet::Model & mod);
  ~SoftmaxMultiLayer() { };

  // Create a new graph
  virtual void NewGraph(dynet::ComputationGraph & cg) override;

  // Calculate training loss for one word
  virtual dynet::Expression CalcLoss(dynet::Expression & in, dynet::Expression & prior, const Sentence & ngram, bool train) override;
  // Calculate training loss for multiple words
  virtual dynet::Expression CalcLoss(dynet::Expression & in, dynet::Expression & prior, const std::vector<Sentence> & ngrams, bool train) override;
  
  // Calculate the full probability distribution
  virtual dynet::Expression CalcProb(dynet::Expression & in, dynet::Expression & prior, const Sentence & ctxt, bool train) override;
  virtual dynet::Expression CalcProb(dynet::Expression & in, dynet::Expression & prior, const std::vector<Sentence> & ctxt, bool train) override;
  virtual dynet::Expression CalcLogProb(dynet::Expression & in, dynet::Expression & prior, const Sentence & ctxt, bool train) override;
  virtual dynet::Expression CalcLogProb(dynet::Expression & in, dynet::Expression & prior, const std::vector<Sentence> & ctxt, bool train) override;

protected:
  dynet::Parameter p_sm_W_; // Softmax weights
  dynet::Parameter p_sm_b_; // Softmax bias

  SoftmaxPtr softmax_;

  dynet::Expression i_sm_W_;
  dynet::Expression i_sm_b_;

};

}
