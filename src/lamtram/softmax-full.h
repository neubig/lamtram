#pragma once

#include <cnn/expr.h>
#include <lamtram/softmax-base.h>

namespace cnn { class Parameters; }

namespace lamtram {

// An interface to a class that takes a vector as input
// (potentially batched) and calculates a probability distribution
// over words
class SoftmaxFull : public SoftmaxBase {

public:
  SoftmaxFull(const std::string & sig, int input_size, const DictPtr & vocab, cnn::Model & mod);
  ~SoftmaxFull() { };

  // Create a new graph
  virtual void NewGraph(cnn::ComputationGraph & cg) override;

  // Calculate training loss for one word
  virtual cnn::expr::Expression CalcLoss(cnn::expr::Expression & in, WordId word, bool train) override;
  // Calculate training loss for multiple words
  virtual cnn::expr::Expression CalcLoss(cnn::expr::Expression & in, const std::vector<WordId> & word, bool train) override;
  
  // Calculate the full probability distribution
  virtual cnn::expr::Expression CalcProbability(cnn::expr::Expression & in) override;
  virtual cnn::expr::Expression CalcLogProbability(cnn::expr::Expression & in) override;

protected:
  cnn::Parameters* p_sm_W_; // Softmax weights
  cnn::Parameters* p_sm_b_; // Softmax bias

  cnn::expr::Expression i_sm_W_;
  cnn::expr::Expression i_sm_b_;

};

}
