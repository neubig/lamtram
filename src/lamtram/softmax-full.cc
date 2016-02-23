#include <lamtram/softmax-full.h>
#include <lamtram/macros.h>
#include <cnn/expr.h>
#include <cnn/dict.h>

using namespace lamtram;
using namespace cnn::expr;
using namespace std;

SoftmaxFull::SoftmaxFull(const std::string & sig, int input_size, const DictPtr & vocab, cnn::Model & mod) : SoftmaxBase(sig,input_size,vocab,mod) {
  p_sm_W_ = mod.add_parameters({(unsigned int)vocab->size(), (unsigned int)input_size});
  p_sm_b_ = mod.add_parameters({(unsigned int)vocab->size()});  
}

void SoftmaxFull::NewGraph(cnn::ComputationGraph & cg) {
  i_sm_b_ = parameter(cg, p_sm_b_);
  i_sm_W_ = parameter(cg, p_sm_W_);
}

// Calculate training loss for one word
cnn::expr::Expression SoftmaxFull::CalcLoss(cnn::expr::Expression & in, const Sentence & ngram, bool train) {
  Expression score = affine_transform({i_sm_b_, i_sm_W_, in});
  return pickneglogsoftmax(score, *ngram.rbegin());
}
// Calculate training loss for multiple words
cnn::expr::Expression SoftmaxFull::CalcLoss(cnn::expr::Expression & in, const std::vector<Sentence> & ngrams, bool train) {
  Expression score = affine_transform({i_sm_b_, i_sm_W_, in});
  std::vector<unsigned> wvec(ngrams.size());
  for(size_t i = 0; i < ngrams.size(); i++)
    wvec[i] = *ngrams[i].rbegin();
  return pickneglogsoftmax(score, wvec);
}

// Calculate the full probability distribution
cnn::expr::Expression SoftmaxFull::CalcProbability(cnn::expr::Expression & in, const Sentence & ctxt) {
  return softmax(affine_transform({i_sm_b_, i_sm_W_, in}));
}
cnn::expr::Expression SoftmaxFull::CalcProbability(cnn::expr::Expression & in, const vector<Sentence> & ctxt) {
  return softmax(affine_transform({i_sm_b_, i_sm_W_, in}));
}
cnn::expr::Expression SoftmaxFull::CalcLogProbability(cnn::expr::Expression & in, const Sentence & ctxt) {
  return log_softmax(affine_transform({i_sm_b_, i_sm_W_, in}));
}
cnn::expr::Expression SoftmaxFull::CalcLogProbability(cnn::expr::Expression & in, const vector<Sentence> & ctxt) {
  return log_softmax(affine_transform({i_sm_b_, i_sm_W_, in}));
}

