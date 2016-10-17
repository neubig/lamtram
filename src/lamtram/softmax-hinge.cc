#include <lamtram/softmax-hinge.h>
#include <lamtram/macros.h>
#include <lamtram/string-util.h>
#include <dynet/expr.h>
#include <dynet/dict.h>

using namespace lamtram;
using namespace dynet::expr;
using namespace std;

SoftmaxHinge::SoftmaxHinge(const std::string & sig, int input_size, const DictPtr & vocab, dynet::Model & mod) : SoftmaxBase(sig,input_size,vocab,mod), margin_(1.f) {
  vector<string> strs = Tokenize(sig, ":");
  if(strs.size() <= 2 || strs[0] != "hinge") THROW_ERROR("Bad signature in SoftmaxHinge: " << sig);
  vector<string> my_dist_files;
  // Read the arguments
  for(size_t i = 1; i < strs.size(); i++) {
    if(strs[i].substr(0, 7) == "margin=") {
      margin_ = stof(strs[i].substr(7));
    } else {
      THROW_ERROR("Illegal option in SoftmaxHinge initializer: " << strs[i]);
    }
  }
  p_sm_W_ = mod.add_parameters({(unsigned int)vocab->size(), (unsigned int)input_size});
  p_sm_b_ = mod.add_parameters({(unsigned int)vocab->size()});  
}

void SoftmaxHinge::NewGraph(dynet::ComputationGraph & cg) {
  i_sm_b_ = parameter(cg, p_sm_b_);
  i_sm_W_ = parameter(cg, p_sm_W_);
}

// Calculate training loss for one word
dynet::expr::Expression SoftmaxHinge::CalcLoss(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ngram, bool train) {
  Expression score = affine_transform({i_sm_b_, i_sm_W_, in});
  if(prior.pg != nullptr) score = score + prior;
  return hinge(score, *ngram.rbegin(), margin_);
}
// Calculate training loss for multiple words
dynet::expr::Expression SoftmaxHinge::CalcLoss(dynet::expr::Expression & in, dynet::expr::Expression & prior, const std::vector<Sentence> & ngrams, bool train) {
  Expression score = affine_transform({i_sm_b_, i_sm_W_, in});
  if(prior.pg != nullptr) score = score + prior;
  std::vector<unsigned> wvec(ngrams.size());
  for(size_t i = 0; i < ngrams.size(); i++)
    wvec[i] = *ngrams[i].rbegin();
  return hinge(score, wvec, margin_);
}

// Calculate the full probability distribution
dynet::expr::Expression SoftmaxHinge::CalcProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ctxt, bool train) {
  THROW_ERROR("CalcProb not valid for hinge");
  return softmax(affine_transform({i_sm_b_, i_sm_W_, in}));
}
dynet::expr::Expression SoftmaxHinge::CalcProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const vector<Sentence> & ctxt, bool train) {
  THROW_ERROR("CalcProb not valid for hinge");
  return softmax(affine_transform({i_sm_b_, i_sm_W_, in}));
}
dynet::expr::Expression SoftmaxHinge::CalcLogProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ctxt, bool train) {
  return (prior.pg != nullptr ?
          affine_transform({i_sm_b_, i_sm_W_, in}) + prior :
          affine_transform({i_sm_b_, i_sm_W_, in}));
}
dynet::expr::Expression SoftmaxHinge::CalcLogProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const vector<Sentence> & ctxt, bool train) {
  return (prior.pg != nullptr ?
          affine_transform({i_sm_b_, i_sm_W_, in}) + prior :
          affine_transform({i_sm_b_, i_sm_W_, in}));
}

