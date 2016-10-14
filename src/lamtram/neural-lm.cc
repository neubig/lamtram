#include <lamtram/neural-lm.h>
#include <lamtram/macros.h>
#include <lamtram/builder-factory.h>
#include <lamtram/extern-calculator.h>
#include <lamtram/softmax-factory.h>
#include <dynet/dict.h>
#include <dynet/model.h>
#include <dynet/nodes.h>
#include <dynet/rnn.h>
#include <boost/range/irange.hpp>
#include <ctime>
#include <fstream>

using namespace std;
using namespace lamtram;

inline std::string print_vec(const std::vector<float> & vec) {
  ostringstream oss;
  if(vec.size() > 0) oss << vec[0];
  for(size_t i = 1; i < vec.size(); ++i) oss << ' ' << vec[i];
  return oss.str();
}

NeuralLM::NeuralLM(const DictPtr & vocab, int ngram_context, int extern_context, bool extern_feed,
           int wordrep_size, const BuilderSpec & hidden_spec, int unk_id, const std::string & softmax_sig,
           dynet::Model & model) :
      vocab_(vocab), ngram_context_(ngram_context),
      extern_context_(extern_context), extern_feed_(extern_feed), wordrep_size_(wordrep_size),
      unk_id_(unk_id), hidden_spec_(hidden_spec), curr_graph_(NULL) {
  if(wordrep_size_ <= 0) wordrep_size_ = GlobalVars::layer_size;
  // Hidden layers
  builder_ = BuilderFactory::CreateBuilder(hidden_spec_,
                       ngram_context*wordrep_size_ + (extern_feed ? extern_context : 0),
                       model);
  // Word representations
  p_wr_W_ = model.add_lookup_parameters(vocab->size(), {(unsigned int)wordrep_size_}); 

  // Create the softmax
  softmax_ = SoftmaxFactory::CreateSoftmax(softmax_sig, hidden_spec_.nodes + extern_context, vocab, model);
}

dynet::expr::Expression NeuralLM::BuildSentGraph(
                      const Sentence & sent,
                      const Sentence & cache_ids,
                      const float * weight,
                      const ExternCalculator * extern_calc,
                      const std::vector<dynet::expr::Expression> & layer_in,
                      float samp_prob,
                      bool train,
                      dynet::ComputationGraph & cg,
                      LLStats & ll) {
  if(samp_prob != 0.f)
    THROW_ERROR("Single-sentence scheduled sampling not implemented yet");
  size_t i;
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
  int slen = sent.size() - 1;
  builder_->start_new_sequence(layer_in);
  // First get all the word representations
  dynet::expr::Expression i_wr_start = lookup(cg, p_wr_W_, (unsigned)0);
  // cerr << "i_wr_start = " << i_wr_start.value().d << endl;
  vector<dynet::expr::Expression> i_wr(slen);
  for(auto t : boost::irange(0, slen))
    i_wr[t] = lookup(cg, p_wr_W_, sent[t]);
  // Initialize the previous extern
  dynet::expr::Expression extern_in;
  assert(extern_context_ == 0 || extern_calc != nullptr);
  if(extern_context_ != 0 && extern_feed_) extern_in = extern_calc->GetEmptyContext(cg);
  // Next, do the computation
  vector<dynet::expr::Expression> errs, aligns;
  dynet::expr::Expression align_sum;
  Sentence ngram(softmax_->GetCtxtLen()+1, 0);
  for(auto t : boost::irange(0, slen+1)) {
    // Concatenate wordrep and possibly external context into a vector for the hidden unit
    vector<dynet::expr::Expression> i_wrs_t;
    for(auto hist : boost::irange(t - ngram_context_, t)) {
      i_wrs_t.push_back(hist >= 0 ? i_wr[hist] : i_wr_start);
    }
    if(extern_context_ > 0 && extern_feed_)
      i_wrs_t.push_back(extern_in);
    // Concatenate the inputs if necessary
    dynet::expr::Expression i_wr_t;
    if(i_wrs_t.size() > 1) {
      i_wr_t = concatenate(i_wrs_t);
    } else {
      assert(i_wrs_t.size() == 1);
      i_wr_t = i_wrs_t[0];
    }
    // cerr << "i_wr_t == " << print_vec(as_vector(i_wr_t.value())) << endl;
    // Run the hidden unit
    dynet::expr::Expression i_h_t = builder_->add_input(i_wr_t);
    dynet::expr::Expression i_prior;
    // Calculate the extern if existing
    if(extern_context_ > 0) {
      extern_in = extern_calc->CreateContext(builder_->final_h(), align_sum, train, cg, aligns, align_sum);
      i_h_t = concatenate({i_h_t, extern_in});
      i_prior = extern_calc->CalcPrior(*aligns.rbegin());
    }
    // If the extern is capable of calculating a probability distribution, do it
    // Run the softmax and calculate the error
    for(i = 0; i < ngram.size()-1; i++) ngram[i] = ngram[i+1];
    ngram[i] = sent[t];
    dynet::expr::Expression i_err = (cache_ids.size() ?
      softmax_->CalcLossCache(i_h_t, i_prior, cache_ids[t], ngram, train) :
      softmax_->CalcLoss(i_h_t, i_prior, ngram, train));
    // DEBUG cerr << ' ' << as_scalar(i_err.value());
    errs.push_back(i_err);
    // If this word is unknown, then add to the unknown count
    if(sent[t] == unk_id_)
      ll.unk_++;
    ll.words_++;
  }
  // DEBUG cerr << endl; 
  dynet::expr::Expression i_nerr = sum(errs);
  // For a single sentence, we can just multiply final error by weight
  if (weight)
    i_nerr = i_nerr * (*weight);
  return i_nerr;
}

inline size_t categorical_dist(std::vector<float>::const_iterator beg, std::vector<float>::const_iterator end) {
  float sum = 0.f;
  for(auto it = beg; it != end; it++)
    sum += *it;
  std::uniform_real_distribution<float> dist(0.f,sum);
  sum = dist(*dynet::rndeng);
  for(auto it = beg; it != end; it++) {
    sum -= *it;
    if(sum < 0) return it-beg;
  }
  cerr << "WARNING: overflowed sample, returning zero" << endl;
  return 0;
}

dynet::expr::Expression NeuralLM::BuildSentGraph(
                      const vector<Sentence> & sent,
                      const vector<Sentence> & cache_ids,
                      const vector<float> * weights,
                      const ExternCalculator * extern_calc,
                      const std::vector<dynet::expr::Expression> & layer_in,
                      float samp_prob,
                      bool train,
                      dynet::ComputationGraph & cg,
                      LLStats & ll) {
  if(sent.size() == 1)
    return BuildSentGraph(sent[0], (cache_ids.size() ? cache_ids[0] : Sentence()), (weights ? &weights->at(0) : nullptr), extern_calc, layer_in, samp_prob, train, cg, ll);
  size_t nt;
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
  int slen = sent[0].size() - 1;
  builder_->start_new_sequence(layer_in);
  // First get all the word representations
  vector<unsigned> words(sent.size(), 0);
  dynet::expr::Expression i_wr_start = lookup(cg, p_wr_W_, words);
  vector<dynet::expr::Expression> i_wr;
  Sentence my_cache(sent.size());
  std::bernoulli_distribution samp_dist(samp_prob);
  // Initialize the previous extern
  dynet::expr::Expression extern_in;
  if(extern_context_ != 0 && extern_feed_) extern_in = extern_calc->GetEmptyContext(cg);
  // Next, do the computation
  vector<dynet::expr::Expression> errs, aligns;
  dynet::expr::Expression align_sum;
  vector<Sentence> ngrams(sent.size(), Sentence(softmax_->GetCtxtLen()+1, 0));
  vector<float> mask(sent.size(), 1.0);
  size_t active_words = sent.size();
  for(auto t : boost::irange(0, slen+1)) {
    // Concatenate wordrep and external context into a vector for the hidden unit
    vector<dynet::expr::Expression> i_wrs_t;
    for(auto hist : boost::irange(t - ngram_context_, t))
      i_wrs_t.push_back(hist >= 0 ? i_wr[hist] : i_wr_start);
    if(extern_context_ > 0 && extern_feed_)
      i_wrs_t.push_back(extern_in);
    // Concatenate the inputs if necessary
    dynet::expr::Expression i_wr_t;
    if(i_wrs_t.size() > 1) {
      i_wr_t = concatenate(i_wrs_t);
    } else {
      assert(i_wrs_t.size() == 1);
      i_wr_t = i_wrs_t[0];
    }
    // Run the hidden unit
    dynet::expr::Expression i_h_t = builder_->add_input(i_wr_t);
    dynet::expr::Expression i_prior;
    // Calculate the extern if existing
    if(extern_context_ > 0) {
      extern_in = extern_calc->CreateContext(builder_->final_h(), align_sum, train, cg, aligns, align_sum);
      i_h_t = concatenate({i_h_t, extern_in});
      i_prior = extern_calc->CalcPrior(*aligns.rbegin());
    }
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
    dynet::expr::Expression i_err;
    if(cache_ids.size()) {
      assert(cache_ids.size() == sent.size());
      for(size_t i = 0; i < sent.size(); i++)
        my_cache[i] = (cache_ids[i].size() > t ? cache_ids[i][t] : 0);
    } 
    // Perform sampling if necessary
    if(samp_dist(*dynet::rndeng)) {
      vector<Sentence> ctxts(ngrams);
      for(size_t i = 0; i < ngrams.size(); i++) {
        words[i] = *ngrams[i].rbegin();
        ctxts[i].resize(ctxts[i].size()-1);
      }
      dynet::expr::Expression i_prob = (
        cache_ids.size() ?
        softmax_->CalcProbCache(i_h_t, i_prior, my_cache, ctxts, train) :
        softmax_->CalcProb(i_h_t, i_prior, ctxts, train));
      i_err = -log(pick(i_prob, words));
      vector<float> probs = as_vector(i_prob.value());
      for(size_t i = 0; i < ngrams.size(); i++)
        words[i] = categorical_dist(probs.begin()+i*vocab_->size(), probs.begin()+(i+1)*vocab_->size());
    // Otherwise, create the correct
    } else {
      i_err = (
        cache_ids.size() ?
        softmax_->CalcLossCache(i_h_t, i_prior, my_cache, ngrams, train) :
        softmax_->CalcLoss(i_h_t, i_prior, ngrams, train));
      for(size_t i = 0; i < sent.size(); i++)
        words[i] = (sent[i].size() > t ? sent[i][t] : 0);
    }
    i_wr.push_back(lookup(cg, p_wr_W_, words));
    if(active_words != sent.size())
      i_err = i_err * input(cg, dynet::Dim({1}, sent.size()), mask);
    // If using weights, weigh error for each sentence's word by that sentence's weight
    if (weights)
      i_err = i_err * input(cg, dynet::Dim({1}, weights->size()), *weights);
    errs.push_back(i_err);
  }
  dynet::expr::Expression i_nerr = sum_batches(sum(errs));
  return i_nerr;
}

// Acquire samples from this sentence and return their log probabilities as a vector
Expression NeuralLM::SampleTrgSentences(
                        const ExternCalculator * extern_calc,
                        const std::vector<dynet::expr::Expression> & layer_in,
                        const Sentence * answer,
                        int num_samples,
                        int max_len,
                        bool train,
                        dynet::ComputationGraph & cg,
                        std::vector<Sentence> & samples) {
  size_t nt;
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
  builder_->start_new_sequence(layer_in);
  // First get all the word representations
  vector<unsigned> words(num_samples, 0);
  dynet::expr::Expression i_wr_start = lookup(cg, p_wr_W_, words);
  vector<dynet::expr::Expression> i_wr;
  // Initialize the previous extern
  dynet::expr::Expression extern_in;
  if(extern_context_ != 0 && extern_feed_) extern_in = extern_calc->GetEmptyContext(cg);
  // Next, do the computation
  vector<dynet::expr::Expression> log_probs, aligns;
  dynet::expr::Expression align_sum;
  vector<Sentence> ctxts(num_samples, Sentence(softmax_->GetCtxtLen(), 0));
  samples.clear(); samples.resize(num_samples);
  vector<float> mask(num_samples, 1.0);
  size_t active_words = num_samples;
  for(int t = 0; t < max_len && active_words > 0; t++) {
    // Concatenate wordrep and external context into a vector for the hidden unit
    vector<dynet::expr::Expression> i_wrs_t;
    for(auto hist : boost::irange(t - ngram_context_, t))
      i_wrs_t.push_back(hist >= 0 ? i_wr[hist] : i_wr_start);
    if(extern_context_ > 0 && extern_feed_)
      i_wrs_t.push_back(extern_in);
    // Concatenate the inputs if necessary
    dynet::expr::Expression i_wr_t;
    if(i_wrs_t.size() > 1) {
      i_wr_t = concatenate(i_wrs_t);
    } else {
      assert(i_wrs_t.size() == 1);
      i_wr_t = i_wrs_t[0];
    }
    // Run the hidden unit
    dynet::expr::Expression i_h_t = builder_->add_input(i_wr_t);
    // Calculate the extern if existing
    dynet::expr::Expression i_prior;
    if(extern_context_ > 0) {
      extern_in = extern_calc->CreateContext(builder_->final_h(), align_sum, train, cg, aligns, align_sum);
      i_h_t = concatenate({i_h_t, extern_in});
      i_prior = extern_calc->CalcPrior(*aligns.rbegin());
    }
    // Create the cache
    dynet::expr::Expression i_err;
    // Perform sampling if necessary
    dynet::expr::Expression i_prob = softmax_->CalcProb(i_h_t, i_prior, ctxts, train);
    vector<float> probs = as_vector(i_prob.value());
    if(mask[0]) {
      if(answer != NULL && t < (int)answer->size())
        words[0] = (*answer)[t];
      else
        words[0] = categorical_dist(probs.begin(), probs.begin()+vocab_->size());
    }
    for(size_t i = 1; i < ctxts.size(); i++)
      if(mask[i])
        words[i] = categorical_dist(probs.begin()+i*vocab_->size(), probs.begin()+(i+1)*vocab_->size());
    // Get the word representations
    i_wr.push_back(lookup(cg, p_wr_W_, words));
    dynet::expr::Expression i_log_pick = log(pick(i_prob, words));
    if(active_words != num_samples)
      i_log_pick = i_log_pick * input(cg, dynet::Dim({1}, num_samples), mask);
    log_probs.push_back(i_log_pick);
    // Update the n-grams
    for(size_t i = 0; i < num_samples; i++) {
      // Count words, add mask
      if(mask[i]) {
        if(ctxts[0].size() > 0) {    
          for(nt = 0; nt < ctxts[0].size()-1; nt++)
            ctxts[i][nt] = ctxts[i][nt+1];
          ctxts[i][nt] = words[i];
        }
        samples[i].push_back(words[i]);
        if(words[i] == 0) {
          mask[i] = 0.f;
          active_words--;
        }
      }
    }
  }
  return reshape(sum(log_probs), dynet::Dim({(unsigned int)num_samples}));
}

void NeuralLM::NewGraph(dynet::ComputationGraph & cg) {
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
dynet::expr::Expression NeuralLM::Forward(const Sent & sent, int t, 
                   const ExternCalculator * extern_calc,
                   bool log_prob,
                   const std::vector<dynet::expr::Expression> & layer_in,
                   const dynet::expr::Expression & extern_in,
                   const dynet::expr::Expression & align_sum_in,
                   std::vector<dynet::expr::Expression> & layer_out,
                   dynet::expr::Expression & extern_out,
                   dynet::expr::Expression & align_sum_out,
                   dynet::ComputationGraph & cg,
                   std::vector<dynet::expr::Expression> & align_out) {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
  // Start a new sequence if necessary
  if(layer_in.size())
    builder_->start_new_sequence(layer_in);
  // Concatenate wordrep and external context into a vector for the hidden unit
  vector<dynet::expr::Expression> i_wrs_t;
  for(auto hist : boost::irange(t - ngram_context_, t))
    i_wrs_t.push_back(lookup(cg, p_wr_W_, CreateWord(sent, hist)));
  if(extern_feed_)
    i_wrs_t.push_back(extern_in.pg == nullptr ? extern_calc->GetEmptyContext(cg) : extern_in);
  // Concatenate the inputs if necessary
  dynet::expr::Expression i_wr_t;
  if(i_wrs_t.size() > 1) {
    i_wr_t = concatenate(i_wrs_t);
  } else {
    assert(i_wrs_t.size() == 1);
    i_wr_t = i_wrs_t[0];
  }
  // cerr << "i_wr_t == " << print_vec(as_vector(i_wr_t.value())) << endl;
  // Run the hidden unit
  dynet::expr::Expression i_h_t = builder_->add_input(i_wr_t);
  dynet::expr::Expression i_prior;
  // Calculate the extern if existing
  if(extern_context_ > 0) {
    extern_out = extern_calc->CreateContext(builder_->final_h(), align_sum_in, false, cg, align_out, align_sum_out);
    i_h_t = concatenate({i_h_t, extern_out});
    i_prior = extern_calc->CalcPrior(*align_out.rbegin());
  }
  // cerr << "i_h_t == " << print_vec(as_vector(i_h_t.value())) << endl;
  // Create the context
  Sent ctxt_ngram = CreateContext<Sent>(sent, t);
  // Run the softmax and calculate the error
  dynet::expr::Expression i_sm_t = (log_prob ?
                  softmax_->CalcLogProb(i_h_t, i_prior, ctxt_ngram, false) :
                  softmax_->CalcProb(i_h_t, i_prior, ctxt_ngram, false));
  // Update the state
  layer_out = builder_->final_s();
  return i_sm_t;
}

// Instantiate
template
dynet::expr::Expression NeuralLM::Forward<Sentence>(
                   const Sentence & sent, int t, 
                   const ExternCalculator * extern_calc,
                   bool log_prob,
                   const std::vector<dynet::expr::Expression> & layer_in,
                   const dynet::expr::Expression & extern_in,
                   const dynet::expr::Expression & sum_in,
                   std::vector<dynet::expr::Expression> & layer_out,
                   dynet::expr::Expression & extern_out,
                   dynet::expr::Expression & sum_out,
                   dynet::ComputationGraph & cg,
                   std::vector<dynet::expr::Expression> & align_out);
template
dynet::expr::Expression NeuralLM::Forward<vector<Sentence> >(
                   const vector<Sentence> & sent, int t, 
                   const ExternCalculator * extern_calc,
                   bool log_prob,
                   const std::vector<dynet::expr::Expression> & layer_in,
                   const dynet::expr::Expression & extern_in,
                   const dynet::expr::Expression & sum_in,
                   std::vector<dynet::expr::Expression> & layer_out,
                   dynet::expr::Expression & extern_out,
                   dynet::expr::Expression & sum_out,
                   dynet::ComputationGraph & cg,
                   std::vector<dynet::expr::Expression> & align_out);

NeuralLM* NeuralLM::Read(const DictPtr & vocab, std::istream & in, dynet::Model & model) {
  int vocab_size, ngram_context, extern_context = 0, wordrep_size, unk_id;
  bool extern_feed;
  string version_id, hidden_spec, line, softmax_sig;
  if(!getline(in, line))
    THROW_ERROR("Premature end of model file when expecting Neural LM");
  istringstream iss(line);
  iss >> version_id;
  if(version_id == "nlm_005") {
    iss >> vocab_size >> ngram_context >> extern_context >> extern_feed >> wordrep_size >> hidden_spec >> unk_id >> softmax_sig;
  } else {
    THROW_ERROR("Expecting a Neural LM of version nlm_005, but got something different:" << endl << line);
  }
  assert(vocab->size() == vocab_size);
  return new NeuralLM(vocab, ngram_context, extern_context, extern_feed, wordrep_size, hidden_spec, unk_id, softmax_sig, model);
}
void NeuralLM::Write(std::ostream & out) {
  out << "nlm_005 " << vocab_->size() << " " << ngram_context_ << " " << extern_context_ << " " << extern_feed_ << " " << wordrep_size_ << " " << hidden_spec_ << " " << unk_id_ << " " << softmax_->GetSig() << endl;
}

int NeuralLM::GetVocabSize() const { return vocab_->size(); }
void NeuralLM::SetDropout(float dropout) { builder_->set_dropout(dropout); }

