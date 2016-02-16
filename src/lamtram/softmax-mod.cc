#include <lamtram/softmax-mod.h>
#include <lamtram/macros.h>
#include <cnn/expr.h>

using namespace lamtram;
using namespace cnn::expr;

SoftmaxMod::SoftmaxMod(const std::string & sig, int input_size, const DictPtr & vocab, cnn::Model & mod) : SoftmaxBase(sig,input_size,vocab,mod) {
  THROW_ERROR("SoftmaxMod::Constructor not implemented yet"); 
}

void SoftmaxMod::NewGraph(cnn::ComputationGraph & cg) {
  THROW_ERROR("SoftmaxMod::NewGraph not implemented yet"); 
}

// Calculate training loss for one word
cnn::expr::Expression SoftmaxMod::CalcLoss(cnn::expr::Expression & in, WordId word, bool train) {
  THROW_ERROR("SoftmaxMod::CalcLoss Not implemented yet");
}
// Calculate training loss for multiple words
cnn::expr::Expression SoftmaxMod::CalcLoss(cnn::expr::Expression & in, const std::vector<WordId> & word, bool train) {
  THROW_ERROR("SoftmaxMod::CalcLoss Not implemented yet");
}

// Calculate the full probability distribution
cnn::expr::Expression SoftmaxMod::CalcProbability(cnn::expr::Expression & in) {
  THROW_ERROR("SoftmaxMod::CalcLogProbability Not implemented yet");
}
cnn::expr::Expression SoftmaxMod::CalcLogProbability(cnn::expr::Expression & in) {
  THROW_ERROR("SoftmaxMod::CalcLogProbability Not implemented yet");
}

