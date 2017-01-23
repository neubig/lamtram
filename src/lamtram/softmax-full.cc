#include <lamtram/softmax-full.h>
#include <lamtram/macros.h>
#include <dynet/expr.h>
#include <dynet/dict.h>

using namespace lamtram;
using namespace dynet::expr;
using namespace std;

SoftmaxFull::SoftmaxFull(const std::string & sig, int input_size, const DictPtr & vocab, dynet::Model & mod) : SoftmaxBase(sig,input_size,vocab,mod) {
  p_sm_W_ = mod.add_parameters({(unsigned int)vocab->size(), (unsigned int)input_size});
  p_sm_b_ = mod.add_parameters({(unsigned int)vocab->size()});  
}

void SoftmaxFull::NewGraph(dynet::ComputationGraph & cg) {
  i_sm_b_ = parameter(cg, p_sm_b_);
  i_sm_W_ = parameter(cg, p_sm_W_);
}

// Calculate training loss for one word
dynet::Expression SoftmaxFull::CalcLoss(dynet::Expression & in, dynet::Expression & prior, const Sentence & ngram, bool train) {
  Expression score = affine_transform({i_sm_b_, i_sm_W_, in});
  if(prior.pg != nullptr) score = score + prior;
  return pickneglogsoftmax(score, *ngram.rbegin());
}
// Calculate training loss for multiple words
dynet::Expression SoftmaxFull::CalcLoss(dynet::Expression & in, dynet::Expression & prior, const std::vector<Sentence> & ngrams, bool train) {
  Expression score = affine_transform({i_sm_b_, i_sm_W_, in});
  if(prior.pg != nullptr) score = score + prior;
  std::vector<unsigned> wvec(ngrams.size());
  for(size_t i = 0; i < ngrams.size(); i++)
    wvec[i] = *ngrams[i].rbegin();
  return pickneglogsoftmax(score, wvec);
}

// Calculate the full probability distribution
dynet::Expression SoftmaxFull::CalcProb(dynet::Expression & in, dynet::Expression & prior, const Sentence & ctxt, bool train) {
  return (prior.pg != nullptr ? 
          softmax(affine_transform({i_sm_b_, i_sm_W_, in}) + prior) :
          softmax(affine_transform({i_sm_b_, i_sm_W_, in})));
}
dynet::Expression SoftmaxFull::CalcProb(dynet::Expression & in, dynet::Expression & prior, const vector<Sentence> & ctxt, bool train) {
  return (prior.pg != nullptr ? 
          softmax(affine_transform({i_sm_b_, i_sm_W_, in}) + prior) :
          softmax(affine_transform({i_sm_b_, i_sm_W_, in})));
}
dynet::Expression SoftmaxFull::CalcLogProb(dynet::Expression & in, dynet::Expression & prior, const Sentence & ctxt, bool train) {
  return (prior.pg != nullptr ? 
          log_softmax(affine_transform({i_sm_b_, i_sm_W_, in}) + prior) :
          log_softmax(affine_transform({i_sm_b_, i_sm_W_, in})));
}
dynet::Expression SoftmaxFull::CalcLogProb(dynet::Expression & in, dynet::Expression & prior, const vector<Sentence> & ctxt, bool train) {
  return (prior.pg != nullptr ? 
          log_softmax(affine_transform({i_sm_b_, i_sm_W_, in}) + prior) :
          log_softmax(affine_transform({i_sm_b_, i_sm_W_, in})));
}

