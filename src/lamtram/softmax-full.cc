#include <lamtram/softmax-full.h>
#include <lamtram/macros.h>
#include <cnn/expr.h>

using namespace lamtram;
using namespace cnn::expr;

SoftmaxFull::SoftmaxFull(const std::string & sig, int input_size, const VocabularyPtr & vocab, cnn::Model & mod) : SoftmaxBase(sig,input_size,vocab,mod) {
  p_sm_W_ = mod.add_parameters({(unsigned int)vocab->size(), (unsigned int)input_size});
  p_sm_b_ = mod.add_parameters({(unsigned int)vocab->size()});  
}

void SoftmaxFull::NewGraph(cnn::ComputationGraph & cg) {
  i_sm_b_ = parameter(cg, p_sm_b_);
  i_sm_W_ = parameter(cg, p_sm_W_);
}

// Calculate training loss for one word
cnn::expr::Expression SoftmaxFull::CalcLoss(cnn::expr::Expression & in, WordId word, bool train) {
  THROW_ERROR("SoftmaxFull::CalcLoss Not implemented yet");
}
// Calculate training loss for multiple words
cnn::expr::Expression SoftmaxFull::CalcLoss(cnn::expr::Expression & in, const std::vector<WordId> & word, bool train) {
  THROW_ERROR("SoftmaxFull::CalcLoss Not implemented yet");
}

// Calculate the full probability distribution
cnn::expr::Expression SoftmaxFull::CalcProbability(cnn::expr::Expression & in) {
  THROW_ERROR("SoftmaxFull::CalcLogProbability Not implemented yet");
}
cnn::expr::Expression SoftmaxFull::CalcLogProbability(cnn::expr::Expression & in) {
  THROW_ERROR("SoftmaxFull::CalcLogProbability Not implemented yet");
}

