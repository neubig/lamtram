#include <lamtram/softmax-mod.h>
#include <lamtram/macros.h>
#include <lamtram/string-util.h>
#include <lamtram/dist-base.h>
#include <lamtram/dist-factory.h>
#include <cnn/expr.h>
#include <cnn/dict.h>

using namespace lamtram;
using namespace cnn::expr;
using namespace std;

SoftmaxMod::SoftmaxMod(const std::string & sig, int input_size, const DictPtr & vocab, cnn::Model & mod) : SoftmaxBase(sig,input_size,vocab,mod) {
  vector<string> strs = Tokenize(sig, ":");
  if(strs.size() <= 2 || strs[0] != "mod") THROW_ERROR("Bad signature in SoftmaxMod: " << sig);
  int num_dist = 0, num_ctxt = 0;
  for(size_t i = 1; i < strs.size(); i++) {
    if(strs[i].substr(0, 8) == "dropout=") {
      dropout_ = stof(strs[i].substr(8));
    } else if(strs[i].substr(0, 5) == "dist=") {
      DistPtr next_ptr = DistFactory::from_file(strs[i].substr(5), vocab);
      if(next_ptr->get_sparse_size() > 0) THROW_ERROR("Sparse distributions not supported yet");
      num_dist += next_ptr->get_dense_size();
      num_ctxt += next_ptr->get_ctxt_size();
      dist_ptrs_.push_back(next_ptr);
    } else {
      THROW_ERROR("Illegal option in SoftmaxMod initializer: " << strs[i]);
    }
  }
  p_sms_W_ = mod.add_parameters({(unsigned int)vocab->size(), (unsigned int)input_size+num_ctxt});
  p_sms_b_ = mod.add_parameters({(unsigned int)vocab->size()});  
  p_smd_W_ = mod.add_parameters({(unsigned int)num_dist, (unsigned int)input_size+num_ctxt});
  p_smd_b_ = mod.add_parameters({(unsigned int)num_dist});  
}

void SoftmaxMod::NewGraph(cnn::ComputationGraph & cg) {
  i_sms_b_ = parameter(cg, p_sms_b_);
  i_sms_W_ = parameter(cg, p_sms_W_);
  i_smd_b_ = parameter(cg, p_smd_b_);
  i_smd_W_ = parameter(cg, p_smd_W_);
}

// Calculate training loss for one word
cnn::expr::Expression SoftmaxMod::CalcLoss(cnn::expr::Expression & in, const Sentence & ngram, bool train) {
  THROW_ERROR("SoftmaxMod::CalcLoss Not implemented yet");
}
// Calculate training loss for multiple words
cnn::expr::Expression SoftmaxMod::CalcLoss(cnn::expr::Expression & in, const std::vector<Sentence> & ngrams, bool train) {
  THROW_ERROR("SoftmaxMod::CalcLoss Not implemented yet");
}

// Calculate the full probability distribution
cnn::expr::Expression SoftmaxMod::CalcProbability(cnn::expr::Expression & in) {
  THROW_ERROR("SoftmaxMod::CalcLogProbability Not implemented yet");
}
cnn::expr::Expression SoftmaxMod::CalcLogProbability(cnn::expr::Expression & in) {
  THROW_ERROR("SoftmaxMod::CalcLogProbability Not implemented yet");
}

