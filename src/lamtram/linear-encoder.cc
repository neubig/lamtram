#include <lamtram/linear-encoder.h>
#include <lamtram/macros.h>
#include <lamtram/builder-factory.h>
#include <cnn/model.h>
#include <cnn/nodes.h>
#include <cnn/rnn.h>
#include <boost/range/irange.hpp>
#include <ctime>
#include <fstream>

using namespace std;
using namespace lamtram;

LinearEncoder::LinearEncoder(int vocab_size, int wordrep_size,
           const string & hidden_spec, int unk_id,
           cnn::Model & model) :
      vocab_size_(vocab_size), wordrep_size_(wordrep_size), unk_id_(unk_id), hidden_spec_(hidden_spec), reverse_(false) {
  // Hidden layers
  builder_ = BuilderFactory::CreateBuilder(hidden_spec_, wordrep_size, model);
  // Word representations
  p_wr_W_ = model.add_lookup_parameters(vocab_size, {(unsigned int)wordrep_size}); 
}

cnn::expr::Expression LinearEncoder::BuildSentGraph(const Sentence & sent, bool add, bool train, cnn::ComputationGraph & cg) {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
  word_states_.resize(sent.size() + (add ? 1 : 0));
  builder_->start_new_sequence();
  // First get all the word representations
  cnn::expr::Expression i_wr_t, i_h_t;
  if(!reverse_) {
    for(int t = 0; t < (int)sent.size(); t++) {
      i_wr_t = lookup(cg, p_wr_W_, sent[t]);
      i_h_t = builder_->add_input(i_wr_t);
      word_states_[t] = i_h_t;
    }
  } else {
    for(int t = sent.size()-1; t >= 0; t--) {
      i_wr_t = lookup(cg, p_wr_W_, sent[t]);
      i_h_t = builder_->add_input(i_wr_t);
      word_states_[t] = i_h_t;
    }
  }
  if(add)
    *word_states_.rbegin() = i_h_t = builder_->add_input(lookup(cg, p_wr_W_, (unsigned)0));
  return i_h_t;
}

cnn::expr::Expression LinearEncoder::BuildSentGraph(const vector<Sentence> & sent, bool add, bool train, cnn::ComputationGraph & cg) {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
  assert(sent.size());
  // Get the max size
  size_t max_len = sent[0].size();
  for(size_t i = 1; i < sent.size(); i++) max_len = max(max_len, sent[i].size());
  if(add) ++max_len;
  // Create the word states
  word_states_.resize(max_len);
  builder_->start_new_sequence();
  // First get all the word representations
  cnn::expr::Expression i_wr_t, i_h_t;
  vector<unsigned> words(sent.size());
  if(!reverse_) {
    for(int t = 0; t < max_len; t++) {
      for(size_t i = 0; i < sent.size(); i++)
        words[i] = (t < sent[i].size() ? sent[i][t] : 0);
      i_wr_t = lookup(cg, p_wr_W_, words);
      i_h_t = builder_->add_input(i_wr_t);
      word_states_[t] = i_h_t;
    }
  } else {
    for(int t = max_len-1; t >= 0; t--) {
      for(size_t i = 0; i < sent.size(); i++)
        words[i] = (t < sent[i].size() ? sent[i][t] : 0);
      i_wr_t = lookup(cg, p_wr_W_, words);
      i_h_t = builder_->add_input(i_wr_t);
      word_states_[t] = i_h_t;
    }
  }
  if(add) {
    std::fill(words.begin(), words.end(), 0);
    *word_states_.rbegin() = i_h_t = builder_->add_input(lookup(cg, p_wr_W_, words));
  }
  return i_h_t;
}


void LinearEncoder::NewGraph(cnn::ComputationGraph & cg) {
  builder_->new_graph(cg);
  curr_graph_ = &cg;
}

// void LinearEncoder::CopyParameters(const LinearEncoder & enc) {
//   assert(vocab_size_ == enc.vocab_size_);
//   assert(wordrep_size_ == enc.wordrep_size_);
//   assert(reverse_ == enc.reverse_);
//   p_wr_W_.copy(enc.p_wr_W_);
//   builder_.copy(enc.builder_);
// }

LinearEncoder* LinearEncoder::Read(std::istream & in, cnn::Model & model) {
  int vocab_size, wordrep_size, unk_id;
  string version_id, hidden_spec, line, reverse;
  if(!getline(in, line))
    THROW_ERROR("Premature end of model file when expecting Neural LM");
  istringstream iss(line);
  iss >> version_id >> vocab_size >> wordrep_size >> hidden_spec >> unk_id >> reverse;
  if(version_id != "linenc_001")
    THROW_ERROR("Expecting a Neural LM of version linenc_001, but got something different:" << endl << line);
  LinearEncoder * ret = new LinearEncoder(vocab_size, wordrep_size, hidden_spec, unk_id, model);
  if(reverse == "rev") ret->SetReverse(true);
  return ret;
}
void LinearEncoder::Write(std::ostream & out) {
  out << "linenc_001 " << vocab_size_ << " " << wordrep_size_ << " " << hidden_spec_ << " " << unk_id_ << " " << (reverse_?"rev":"for") << endl;
}

vector<cnn::expr::Expression> LinearEncoder::GetFinalHiddenLayers() const {
  return builder_->final_h();
}

void LinearEncoder::SetDropout(float dropout) { builder_->set_dropout(dropout); }
