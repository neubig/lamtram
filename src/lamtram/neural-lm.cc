#include <lamtram/neural-lm.h>
#include <lamtram/macros.h>
#include <lamtram/builder-factory.h>
#include <lamtram/extern-calculator.h>
#include <lamtram/softmax-factory.h>
#include <cnn/dict.h>
#include <cnn/model.h>
#include <cnn/nodes.h>
#include <cnn/rnn.h>
#include <boost/range/irange.hpp>
#include <ctime>
#include <fstream>

using namespace std;
using namespace lamtram;

NeuralLM::NeuralLM(const DictPtr & vocab, int ngram_context, int extern_context, int wordrep_size,
           const string & hidden_spec, int unk_id, const std::string & softmax_sig,
           cnn::Model & model) :
      vocab_(vocab), ngram_context_(ngram_context),
      extern_context_(extern_context), wordrep_size_(wordrep_size),
      unk_id_(unk_id), hidden_spec_(hidden_spec), curr_graph_(NULL) {
  // Hidden layers
  builder_ = BuilderFactory::CreateBuilder(hidden_spec_,
                       ngram_context*wordrep_size + extern_context,
                       model);
  // Word representations
  p_wr_W_ = model.add_lookup_parameters(vocab->size(), {(unsigned int)wordrep_size}); 

  // Create the softmax
  softmax_ = SoftmaxFactory::CreateSoftmax(softmax_sig, hidden_spec_.nodes, vocab, model);
}

cnn::expr::Expression NeuralLM::BuildSentGraph(
                      const Sentence & sent,
                      const Sentence & cache_ids,
                      const ExternCalculator * extern_calc,
                      const std::vector<cnn::expr::Expression> & layer_in,
                      float samp_prob,
                      bool train,
                      cnn::ComputationGraph & cg, LLStats & ll) {
  // DEBUG if(!train) cerr << "LOSS @ " << sent << ": " << endl;
  size_t i;
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
  int slen = sent.size() - 1;
  builder_->start_new_sequence(layer_in);
  // First get all the word representations
  cnn::expr::Expression i_wr_start = lookup(cg, p_wr_W_, (unsigned)0);
  vector<cnn::expr::Expression> i_wr;
  for(auto t : boost::irange(0, slen))
    i_wr.push_back(lookup(cg, p_wr_W_, sent[t]));
  // Next, do the computation
  vector<cnn::expr::Expression> errs, aligns;
  Sentence ngram(softmax_->GetCtxtLen()+1, 0);
  for(auto t : boost::irange(0, slen+1)) {
    // Concatenate wordrep and external context into a vector for the hidden unit
    vector<cnn::expr::Expression> i_wrs_t;
    for(auto hist : boost::irange(t - ngram_context_, t))
      i_wrs_t.push_back(hist >= 0 ? i_wr[hist] : i_wr_start);
    if(extern_context_ > 0) {
      assert(extern_calc != NULL);
      cnn::expr::Expression extern_in;
      extern_in = extern_calc->CreateContext(builder_->final_h(), train, cg, aligns);
      i_wrs_t.push_back(extern_in);
    }
    cnn::expr::Expression i_wr_t = concatenate(i_wrs_t);
    // Run the hidden unit
    cnn::expr::Expression i_h_t = builder_->add_input(i_wr_t);
    // Run the softmax and calculate the error
    for(i = 0; i < ngram.size()-1; i++) ngram[i] = ngram[i+1];
    ngram[i] = sent[t];
    cnn::expr::Expression i_err = (cache_ids.size() ?
      softmax_->CalcLossCache(i_h_t, cache_ids[t], ngram, train) :
      softmax_->CalcLoss(i_h_t, ngram, train));
    // DEBUG if(!train) cerr << ' ' << as_scalar(i_err.value());
    errs.push_back(i_err);
    // If this word is unknown, then add to the unknown count
    if(sent[t] == unk_id_)
      ll.unk_++;
    ll.words_++;
  }
  // DEBUG if(!train) cerr << endl;
  
  cnn::expr::Expression i_nerr = sum(errs);
  return i_nerr;
}

inline size_t categorical_dist(std::vector<float>::const_iterator beg, std::vector<float>::const_iterator end) {
  float sum = 0.f;
  std::accumulate(beg,end,sum);
  std::uniform_real_distribution<float> dist(sum);
  sum = dist(*cnn::rndeng);
  for(auto it = beg; it != end; it++) {
    sum -= *it;
    if(sum < 0) return it-beg;
  }
  THROW_ERROR("Overflowed sample");
}

cnn::expr::Expression NeuralLM::BuildSentGraph(
                      const vector<Sentence> & sent,
                      const vector<Sentence> & cache_ids,
                      const ExternCalculator * extern_calc,
                      const std::vector<cnn::expr::Expression> & layer_in,
                      float samp_prob,
                      bool train,
                      cnn::ComputationGraph & cg, LLStats & ll) {
  if(sent.size() == 1)
    return BuildSentGraph(sent[0], (cache_ids.size() ? cache_ids[0] : Sentence()), extern_calc, layer_in, samp_prob, train, cg, ll);
  size_t nt;
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
  int slen = sent[0].size() - 1;
  builder_->start_new_sequence(layer_in);
  // First get all the word representations
  vector<unsigned> words(sent.size(), 0);
  cnn::expr::Expression i_wr_start = lookup(cg, p_wr_W_, words);
  vector<cnn::expr::Expression> i_wr;
  Sentence my_cache(sent.size());
  std::bernoulli_distribution samp_dist(samp_prob);
  // Next, do the computation
  vector<cnn::expr::Expression> errs, aligns;
  vector<Sentence> ngrams(sent.size(), Sentence(softmax_->GetCtxtLen()+1, 0));
  vector<float> mask(sent.size(), 1.0);
  size_t active_words = sent.size();
  for(auto t : boost::irange(0, slen+1)) {
    // Concatenate wordrep and external context into a vector for the hidden unit
    vector<cnn::expr::Expression> i_wrs_t;
    for(auto hist : boost::irange(t - ngram_context_, t))
      i_wrs_t.push_back(hist >= 0 ? i_wr[hist] : i_wr_start);
    if(extern_context_ > 0) {
      assert(extern_calc != NULL);
      cnn::expr::Expression extern_in;
      extern_in = extern_calc->CreateContext(builder_->final_h(), train, cg, aligns);
      i_wrs_t.push_back(extern_in);
    }
    cnn::expr::Expression i_wr_t = concatenate(i_wrs_t);
    // Run the hidden unit
    cnn::expr::Expression i_h_t = builder_->add_input(i_wr_t);
    // Run the softmax and calculate the error
    for(size_t i = 0; i < sent.size(); i++) {
      for(nt = 0; nt < ngrams[0].size()-1; nt++)
        ngrams[i][nt] = ngrams[i][nt+1];
      // Count words, add mask
      if(sent[i].size() > t) {
        ll.words_++;
        if(sent[i][t] == unk_id_) ll.unk_++;
        ngrams[i][nt] = sent[i][t];
      } else if(sent[i].size() == t) {
        mask[i] = 0.f;
        active_words--;
        ngrams[i][nt] = 0;
      } else {
        ngrams[i][nt] = 0;
      }
    }
    // Create the cache
    cnn::expr::Expression i_err;
    if(cache_ids.size()) {
      assert(cache_ids.size() == sent.size());
      for(size_t i = 0; i < sent.size(); i++)
        my_cache[i] = (cache_ids[i].size() > t ? cache_ids[i][t] : 0);
    } 
    // Perform sampling if necessary
    if(samp_dist(*cnn::rndeng)) {
      vector<Sentence> ctxts(ngrams);
      for(size_t i = 0; i < ngrams.size(); i++) {
        words[i] = *ngrams[i].rbegin();
        ctxts[i].resize(ctxts[i].size()-1);
      }
      cnn::expr::Expression i_prob = softmax_->CalcProbCache(i_h_t, my_cache, ctxts, train);
      i_err = -log(pick(i_prob, words));
      vector<float> probs = as_vector(i_prob.value());
      for(size_t i = 0; i < ngrams.size(); i++)
        words[i] = categorical_dist(probs.begin()+i*vocab_->size(), probs.begin()+(i+1)*vocab_->size());
    // Otherwise, create the correct
    } else {
      i_err = softmax_->CalcLossCache(i_h_t, my_cache, ngrams, train);
      for(size_t i = 0; i < sent.size(); i++)
        words[i] = (sent[i].size() > t ? sent[i][t] : 0);
    }
    i_wr.push_back(lookup(cg, p_wr_W_, words));
    if(active_words != sent.size())
      i_err = i_err * input(cg, cnn::Dim({1}, sent.size()), mask);
    errs.push_back(i_err);
  }
  cnn::expr::Expression i_nerr = sum_batches(sum(errs));
  return i_nerr;
}

void NeuralLM::NewGraph(cnn::ComputationGraph & cg) {
  builder_->new_graph(cg);
  builder_->start_new_sequence();
  softmax_->NewGraph(cg);
  curr_graph_ = &cg;
}

inline unsigned CreateWord(const Sentence & sent, int t) {
  return (t >= 0 && t < (int)sent.size()) ? sent[t] : 0;
}
inline vector<unsigned> CreateWord(const vector<Sentence> & sent, int t) {
  vector<unsigned> ret(sent.size());
  for(size_t i = 0; i < sent.size(); i++)
    ret[i] = CreateWord(sent[i], t);
  return ret;
}

namespace lamtram {

template <>
Sentence NeuralLM::CreateContext<Sentence>(const Sentence & sent, int t) {
  Sentence ctxt_ngram(softmax_->GetCtxtLen(), 0);
  for(int i = 0, j = t-softmax_->GetCtxtLen(); i < softmax_->GetCtxtLen(); i++, j++)
    if(j >= 0)
      ctxt_ngram[i] = sent[j];
  return ctxt_ngram;
}

template <>
vector<Sentence> NeuralLM::CreateContext<vector<Sentence> >(const vector<Sentence> & sent, int t) {
  vector<Sentence> ret(sent.size());
  for(size_t i = 0; i < sent.size(); i++)
    ret[i] = CreateContext(sent[i], t);
  return ret;
}

}

// Move forward one step using the language model and return the probabilities
template <class Sent>
cnn::expr::Expression NeuralLM::Forward(const Sent & sent, int t, 
                   const ExternCalculator * extern_calc,
                   bool log_prob,
                   const std::vector<cnn::expr::Expression> & layer_in,
                   std::vector<cnn::expr::Expression> & layer_out,
                   cnn::ComputationGraph & cg,
                   std::vector<cnn::expr::Expression> & align_out) {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
  // Start a new sequence if necessary
  if(layer_in.size())
    builder_->start_new_sequence(layer_in);
  // Concatenate wordrep and external context into a vector for the hidden unit
  vector<cnn::expr::Expression> i_wrs_t;
  for(auto hist : boost::irange(t - ngram_context_, t))
    i_wrs_t.push_back(lookup(cg, p_wr_W_, CreateWord(sent, hist)));
  if(extern_context_ > 0) {
    cnn::expr::Expression extern_in = extern_calc->CreateContext(layer_in, false, cg, align_out);
    i_wrs_t.push_back(extern_in);
  }
  cnn::expr::Expression i_wr_t = concatenate(i_wrs_t);
  // Run the hidden unit
  cnn::expr::Expression i_h_t = builder_->add_input(i_wr_t);
  // Create the context
  Sent ctxt_ngram = CreateContext<Sent>(sent, t);
  // Run the softmax and calculate the error
  cnn::expr::Expression i_sm_t = (log_prob ?
                  softmax_->CalcLogProb(i_h_t, ctxt_ngram, false) :
                  softmax_->CalcProb(i_h_t, ctxt_ngram, false));
  // Update the state
  layer_out = builder_->final_s();
  return i_sm_t;
}

// Instantiate
template
cnn::expr::Expression NeuralLM::Forward<Sentence>(
                   const Sentence & sent, int t, 
                   const ExternCalculator * extern_calc,
                   bool log_prob,
                   const std::vector<cnn::expr::Expression> & layer_in,
                   std::vector<cnn::expr::Expression> & layer_out,
                   cnn::ComputationGraph & cg,
                   std::vector<cnn::expr::Expression> & align_out);
template
cnn::expr::Expression NeuralLM::Forward<vector<Sentence> >(
                   const vector<Sentence> & sent, int t, 
                   const ExternCalculator * extern_calc,
                   bool log_prob,
                   const std::vector<cnn::expr::Expression> & layer_in,
                   std::vector<cnn::expr::Expression> & layer_out,
                   cnn::ComputationGraph & cg,
                   std::vector<cnn::expr::Expression> & align_out);

NeuralLM* NeuralLM::Read(const DictPtr & vocab, std::istream & in, cnn::Model & model) {
  int vocab_size, ngram_context, extern_context = 0, wordrep_size, unk_id;
  string version_id, hidden_spec, line, softmax_sig;
  if(!getline(in, line))
    THROW_ERROR("Premature end of model file when expecting Neural LM");
  istringstream iss(line);
  iss >> version_id;
  if(version_id == "nlm_004") {
    iss >> vocab_size >> ngram_context >> extern_context >> wordrep_size >> hidden_spec >> unk_id >> softmax_sig;
  } else {
    THROW_ERROR("Expecting a Neural LM of version nlm_004, but got something different:" << endl << line);
  }
  assert(vocab->size() == vocab_size);
  return new NeuralLM(vocab, ngram_context, extern_context, wordrep_size, hidden_spec, unk_id, softmax_sig, model);
}
void NeuralLM::Write(std::ostream & out) {
  out << "nlm_004 " << vocab_->size() << " " << ngram_context_ << " " << extern_context_ << " " << wordrep_size_ << " " << hidden_spec_ << " " << unk_id_ << " " << softmax_->GetSig() << endl;
}

int NeuralLM::GetVocabSize() const { return vocab_->size(); }

