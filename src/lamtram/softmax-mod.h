#pragma once

#include <cnn/expr.h>
#include <lamtram/softmax-base.h>
#include <lamtram/dist-base.h>

namespace cnn { class Parameters; }

namespace lamtram {

// An interface to a class that takes a vector as input
// (potentially batched) and calculates a probability distribution
// over words
class SoftmaxMod : public SoftmaxBase {

public:
  SoftmaxMod(const std::string & sig, int input_size, const DictPtr & vocab, cnn::Model & mod);
  ~SoftmaxMod() { };

  // Create a new graph
  virtual void NewGraph(cnn::ComputationGraph & cg) override;

  // Calculate training loss for one word
  virtual cnn::expr::Expression CalcLoss(cnn::expr::Expression & in, const Sentence & ngram, bool train) override;
  // Calculate training loss for multiple words
  virtual cnn::expr::Expression CalcLoss(cnn::expr::Expression & in, const std::vector<Sentence> & ngrams, bool train) override;
  
  // Calculate the full probability distribution
  virtual cnn::expr::Expression CalcProbability(cnn::expr::Expression & in) override;
  virtual cnn::expr::Expression CalcLogProbability(cnn::expr::Expression & in) override;

protected:
  cnn::Parameters *p_sms_W_, *p_smd_W_; // Softmax weights
  cnn::Parameters *p_sms_b_, *p_smd_b_; // Softmax bias

  cnn::expr::Expression i_sms_W_, i_smd_W_;
  cnn::expr::Expression i_sms_b_, i_smd_b_;

  float dropout_; // How much to drop out the dense distributions (at training)
  std::vector<DistPtr> dist_ptrs_;

};

}
