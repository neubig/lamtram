#include <lamtram/softmax-class.h>
#include <lamtram/macros.h>
#include <lamtram/string-util.h>
#include <dynet/cfsm-builder.h>
#include <dynet/expr.h>

using namespace lamtram;
using namespace dynet;
using namespace std;

SoftmaxClass::SoftmaxClass(const std::string & sig, int input_size, const DictPtr & vocab, ParameterCollection & mod) : SoftmaxBase(sig,input_size,vocab,mod) {
  vector<string> strs = Tokenize(sig, ":");
  if(strs.size() != 2 || strs[0] != "class") THROW_ERROR("Bad signature in SoftmaxClass: " << sig);
  cfsm_builder_.reset(new ClassFactoredSoftmaxBuilder(input_size, strs[1], *vocab, mod));
}

void SoftmaxClass::NewGraph(ComputationGraph & cg) {
  cfsm_builder_->new_graph(cg);
}

// Calculate training loss for one word
Expression SoftmaxClass::CalcLoss(Expression & in, Expression & prior, const Sentence & ngram, bool train) {
  assert(prior.pg == nullptr);
  return cfsm_builder_->neg_log_softmax(in, *ngram.rbegin());
}
// Calculate training loss for multiple words
Expression SoftmaxClass::CalcLoss(Expression & in, Expression & prior, const std::vector<Sentence> & ngrams, bool train) {
  THROW_ERROR("SoftmaxClass::CalcLoss Not implemented for batches yet");
}

// Calculate the full probability distribution
Expression SoftmaxClass::CalcProb(Expression & in, Expression & prior, const Sentence & ctxt, bool train) {
  THROW_ERROR("SoftmaxClass::CalcProb Not implemented yet");
}
Expression SoftmaxClass::CalcProb(Expression & in, Expression & prior, const vector<Sentence> & ctxt, bool train) {
  THROW_ERROR("SoftmaxClass::CalcProb Not implemented yet");
}
Expression SoftmaxClass::CalcLogProb(Expression & in, Expression & prior, const Sentence & ctxt, bool train) {
  THROW_ERROR("SoftmaxClass::CalcLogProb Not implemented yet");
}
Expression SoftmaxClass::CalcLogProb(Expression & in, Expression & prior, const vector<Sentence> & ctxt, bool train) {
  THROW_ERROR("SoftmaxClass::CalcLogProb Not implemented yet");
}

