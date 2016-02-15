#include <lamtram/softmax-class.h>
#include <lamtram/macros.h>
#include <cnn/expr.h>

using namespace lamtram;
using namespace cnn::expr;

SoftmaxClass::SoftmaxClass(const std::string & sig, int input_size, const VocabularyPtr & vocab, cnn::Model & mod) : SoftmaxBase(sig,input_size,vocab,mod) {
  THROW_ERROR("SoftmaxClass::Constructor not implemented yet"); 
}

void SoftmaxClass::NewGraph(cnn::ComputationGraph & cg) {
  THROW_ERROR("SoftmaxClass::NewGraph not implemented yet"); 
}

// Calculate training loss for one word
cnn::expr::Expression SoftmaxClass::CalcLoss(cnn::expr::Expression & in, WordId word, bool train) {
  THROW_ERROR("SoftmaxClass::CalcLoss Not implemented yet");
}
// Calculate training loss for multiple words
cnn::expr::Expression SoftmaxClass::CalcLoss(cnn::expr::Expression & in, const std::vector<WordId> & word, bool train) {
  THROW_ERROR("SoftmaxClass::CalcLoss Not implemented yet");
}

// Calculate the full probability distribution
cnn::expr::Expression SoftmaxClass::CalcProbability(cnn::expr::Expression & in) {
  THROW_ERROR("SoftmaxClass::CalcLogProbability Not implemented yet");
}
cnn::expr::Expression SoftmaxClass::CalcLogProbability(cnn::expr::Expression & in) {
  THROW_ERROR("SoftmaxClass::CalcLogProbability Not implemented yet");
}

