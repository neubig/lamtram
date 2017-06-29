#pragma once

#include <lamtram/softmax-base.h>
#include <dynet/model.h>

namespace dynet {
  class ClassFactoredSoftmaxBuilder;
}

namespace lamtram {


// An interface to a class that takes a vector as input
// (potentially batched) and calculates a probability distribution
// over words
class SoftmaxClass : public SoftmaxBase {

public:
  SoftmaxClass(const std::string & sig, int input_size, const DictPtr & vocab, dynet::ParameterCollection & mod);
  ~SoftmaxClass() { };

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
  std::shared_ptr<dynet::ClassFactoredSoftmaxBuilder> cfsm_builder_;

};

}
