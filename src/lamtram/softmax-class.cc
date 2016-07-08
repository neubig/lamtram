#include <lamtram/softmax-class.h>
#include <lamtram/macros.h>
#include <lamtram/string-util.h>
#include <cnn/cfsm-builder.h>
#include <cnn/expr.h>

using namespace lamtram;
using namespace cnn::expr;
using namespace std;

SoftmaxClass::SoftmaxClass(const std::string & sig, int input_size, const DictPtr & vocab, cnn::Model & mod) : SoftmaxBase(sig,input_size,vocab,mod) {
  vector<string> strs = Tokenize(sig, ":");
  if(strs.size() != 2 || strs[0] != "class") THROW_ERROR("Bad signature in SoftmaxClass: " << sig);
  cfsm_builder_.reset(new cnn::ClassFactoredSoftmaxBuilder(input_size, strs[1], vocab.get(), &mod));
}

void SoftmaxClass::NewGraph(cnn::ComputationGraph & cg) {
  cfsm_builder_->new_graph(cg);
}

// Calculate training loss for one word
cnn::expr::Expression SoftmaxClass::CalcLoss(cnn::expr::Expression & in, cnn::expr::Expression & prior, const Sentence & ngram, bool train) {
  assert(prior.pg == nullptr);
  return cfsm_builder_->neg_log_softmax(in, *ngram.rbegin());
}
// Calculate training loss for multiple words
cnn::expr::Expression SoftmaxClass::CalcLoss(cnn::expr::Expression & in, cnn::expr::Expression & prior, const std::vector<Sentence> & ngrams, bool train) {
  THROW_ERROR("SoftmaxClass::CalcLoss Not implemented for batches yet");
}

// Calculate the full probability distribution
cnn::expr::Expression SoftmaxClass::CalcProb(cnn::expr::Expression & in, cnn::expr::Expression & prior, const Sentence & ctxt, bool train) {
  THROW_ERROR("SoftmaxClass::CalcProb Not implemented yet");
}
cnn::expr::Expression SoftmaxClass::CalcProb(cnn::expr::Expression & in, cnn::expr::Expression & prior, const vector<Sentence> & ctxt, bool train) {
  THROW_ERROR("SoftmaxClass::CalcProb Not implemented yet");
}
cnn::expr::Expression SoftmaxClass::CalcLogProb(cnn::expr::Expression & in, cnn::expr::Expression & prior, const Sentence & ctxt, bool train) {
  THROW_ERROR("SoftmaxClass::CalcLogProb Not implemented yet");
}
cnn::expr::Expression SoftmaxClass::CalcLogProb(cnn::expr::Expression & in, cnn::expr::Expression & prior, const vector<Sentence> & ctxt, bool train) {
  THROW_ERROR("SoftmaxClass::CalcLogProb Not implemented yet");
}

