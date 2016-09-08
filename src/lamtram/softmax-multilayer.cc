#include <lamtram/softmax-multilayer.h>
#include <lamtram/macros.h>
#include <cnn/expr.h>
#include <cnn/dict.h>

using namespace lamtram;
using namespace cnn::expr;
using namespace std;

SoftmaxMultiLayer::SoftmaxMultiLayer(const std::string & sig, int input_size, const DictPtr & vocab, cnn::Model & mod) : SoftmaxBase(sig,input_size,vocab,mod) {

  vector<string> strs;
    boost::algorithm::split(strs, sig, boost::is_any_of(":"));

  int hiddenSize = atoi(strs[1].c_str());
  p_sm_W_ = mod.add_parameters({(unsigned int)hiddenSize, (unsigned int)input_size});
  p_sm_b_ = mod.add_parameters({(unsigned int)hiddenSize});
  strs.erase(strs.begin(), strs.begin() + 2);
  std::string new_sig = boost::algorithm::join(strs, ":");
  softmax_ = SoftmaxFactory::CreateSoftmax(new_sig, hiddenSize, vocab, mod);
}

void SoftmaxMultiLayer::NewGraph(cnn::ComputationGraph & cg) {
  i_sm_b_ = parameter(cg, p_sm_b_);
  i_sm_W_ = parameter(cg, p_sm_W_);
  softmax_->NewGraph(cg);
}

// Calculate training loss for one word
cnn::expr::Expression SoftmaxMultiLayer::CalcLoss(cnn::expr::Expression & in, cnn::expr::Expression & prior, const Sentence & ngram, bool train) {
  Expression score = affine_transform({i_sm_b_, i_sm_W_, in});
  score = tanh(score);
  return softmax_->CalcLoss(score,prior,ngram,train);
}
// Calculate training loss for multiple words
cnn::expr::Expression SoftmaxMultiLayer::CalcLoss(cnn::expr::Expression & in, cnn::expr::Expression & prior, const std::vector<Sentence> & ngrams, bool train) {
  Expression score = affine_transform({i_sm_b_, i_sm_W_, in});
  score = tanh(score);
  return softmax_->CalcLoss(score,prior,ngrams,train);
}

// Calculate the full probability distribution
cnn::expr::Expression SoftmaxMultiLayer::CalcProb(cnn::expr::Expression & in, cnn::expr::Expression & prior, const Sentence & ctxt, bool train) {
  cnn::expr::Expression h = tanh(affine_transform({i_sm_b_, i_sm_W_, in}));
  return softmax_->CalcProb(h,prior,ctxt,train);
}
cnn::expr::Expression SoftmaxMultiLayer::CalcProb(cnn::expr::Expression & in, cnn::expr::Expression & prior, const vector<Sentence> & ctxt, bool train) {
  cnn::expr::Expression h = tanh(affine_transform({i_sm_b_, i_sm_W_, in}));
  return softmax_->CalcProb(h,prior,ctxt,train);
}
cnn::expr::Expression SoftmaxMultiLayer::CalcLogProb(cnn::expr::Expression & in, cnn::expr::Expression & prior, const Sentence & ctxt, bool train) {
  cnn::expr::Expression h = tanh(affine_transform({i_sm_b_, i_sm_W_, in}));
  return softmax_->CalcLogProb(h,prior,ctxt,train);
}
cnn::expr::Expression SoftmaxMultiLayer::CalcLogProb(cnn::expr::Expression & in, cnn::expr::Expression & prior, const vector<Sentence> & ctxt, bool train) {
  cnn::expr::Expression h = tanh(affine_transform({i_sm_b_, i_sm_W_, in}));
  return softmax_->CalcLogProb(h,prior,ctxt,train);
}

