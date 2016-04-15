#include <lamtram/softmax-mod.h>
#include <lamtram/macros.h>
#include <lamtram/string-util.h>
#include <lamtram/dist-base.h>
#include <lamtram/dist-factory.h>
#include <lamtram/hashes.h>
#include <cnn/expr.h>
#include <cnn/dict.h>

using namespace lamtram;
using namespace cnn::expr;
using namespace std;

void SoftmaxMod::LoadDists(int id) {
  if(dist_id_ == id) return;
  dist_ptrs_.resize(dist_files_.size());
  for(size_t i = 0; i < dist_files_.size(); i++) {
    if(dist_files_[i].size() != 1) {
      assert(id < (int)dist_files_[i].size());
      cerr << "Loading distribution: " << dist_files_[i][id] << "..." << endl;
      dist_ptrs_[i] = DistFactory::from_file(dist_files_[i][id], vocab_);
    } else if(dist_ptrs_[i].get() != NULL) {
      cerr << "Loading distribution: " << dist_files_[i][0] << "..." << endl;
      dist_ptrs_[i] = DistFactory::from_file(dist_files_[i][0], vocab_);
    }
    if(dist_ptrs_[i]->get_sparse_size() > 0) THROW_ERROR("Sparse distributions not supported yet");
  }
  dist_id_ = id;
}

SoftmaxMod::SoftmaxMod(const string & sig, int input_size, const DictPtr & vocab, cnn::Model & mod) : SoftmaxBase(sig,input_size,vocab,mod), num_dist_(0), num_ctxt_(0), finished_words_(0), drop_words_(0), dist_id_(-1) {
  vector<string> strs = Tokenize(sig, ":");
  if(strs.size() <= 2 || strs[0] != "mod") THROW_ERROR("Bad signature in SoftmaxMod: " << sig);
  vector<string> my_dist_files;
  // Read the arguments
  for(size_t i = 1; i < strs.size(); i++) {
    if(strs[i].substr(0, 8) == "dropout=") {
      dropout_ = stof(strs[i].substr(8));
    } else if(strs[i].substr(0, 10) == "dropwords=") {
      drop_words_ = stof(strs[i].substr(10));
    } else if(strs[i].substr(0, 5) == "dist=") {
      my_dist_files.push_back(strs[i].substr(5));
    } else if(strs[i].substr(0, 10) == "wildcards=") {
      wildcards_ = Tokenize(strs[i].substr(10), "|");
    } else {
      THROW_ERROR("Illegal option in SoftmaxMod initializer: " << strs[i]);
    }
  }
  // Read in the distributions
  for(auto & dist_file : my_dist_files)
    dist_files_.push_back(TokenizeWildcarded(dist_file,wildcards_,"|"));
  // Read the distributions for the first set of training data
  LoadDists(0);
  ctxt_len_ = 0;
  for(auto & dist : dist_ptrs_) {
    num_dist_ += dist->get_dense_size();
    num_ctxt_ += dist->get_ctxt_size();
    ctxt_len_ = std::max(ctxt_len_, (int)dist->get_ctxt_len());
  }
  // Initialize the parameters
  p_sms_W_ = mod.add_parameters({(unsigned int)vocab->size(), (unsigned int)input_size+num_ctxt_});
  p_sms_b_ = mod.add_parameters({(unsigned int)vocab->size()});  
  p_smd_W_ = mod.add_parameters({(unsigned int)num_dist_, (unsigned int)input_size+num_ctxt_});
  p_smd_b_ = mod.add_parameters({(unsigned int)num_dist_});  
}

void SoftmaxMod::NewGraph(cnn::ComputationGraph & cg) {
  i_sms_b_ = parameter(cg, p_sms_b_);
  i_sms_W_ = parameter(cg, p_sms_W_);
  i_smd_b_ = parameter(cg, p_smd_b_);
  i_smd_W_ = parameter(cg, p_smd_W_);
}

// Calculate the context and distribution for the current ngram
void SoftmaxMod::CalcDists(const Sentence & ngram, CtxtDist & ctxt_dist) {
  vector<pair<WordId,float> > sparse_dist;
  int dense_offset = 0, sparse_offset = 0, ctxt_offset = 0;
  Sentence ctxt_ngram(ngram); ctxt_ngram.resize(ngram.size()-1);
  for(auto & dist : dist_ptrs_) {
    dist->calc_word_dists(ngram, 1.f/vocab_->size(), 1.f, ctxt_dist.second, dense_offset, sparse_dist, sparse_offset);
    dist->calc_ctxt_feats(ctxt_ngram, &ctxt_dist.first[ctxt_offset]);
    ctxt_offset += dist->get_ctxt_size();
  }
}
void SoftmaxMod::CalcAllDists(const Sentence & ctxt_ngram, CtxtDist & ctxt_dist) {
  vector<pair<pair<int,int>,float> > sparse_dist;
  int dense_offset = 0, sparse_offset = 0, ctxt_offset = 0;
  for(auto & dist : dist_ptrs_) {
    dist->calc_all_word_dists(ctxt_ngram, vocab_->size(), 1.f/vocab_->size(), 1.f, ctxt_dist.second, dense_offset, sparse_dist, sparse_offset);
    dist->calc_ctxt_feats(ctxt_ngram, &ctxt_dist.first[ctxt_offset]);
    ctxt_offset += dist->get_ctxt_size();
  }
}

void SoftmaxMod::Cache(const vector<Sentence> & sents, const vector<int> & set_ids, vector<Sentence> & cache_ids) {
  assert(sents.size() == set_ids.size());
  // Create pairs of context floats, distribution floats
  CtxtDist curr_ctxt_dist; curr_ctxt_dist.first.resize(num_ctxt_); curr_ctxt_dist.second.resize(num_dist_);
  std::unordered_map<CtxtDist, int> ctxt_map;
  size_t i, j, k;
  // Fill the cache with values
  cache_ids.resize(sents.size());
  for(i = 0; i < sents.size(); i++) {
    LoadDists(set_ids[i]+1);
    Sentence ngram(ctxt_len_+1, 0);
    cache_ids[i].resize(sents[i].size());
    for(j = 0; j < sents[i].size(); j++) {
      for(k = 0; k < ngram.size()-1; k++)
        ngram[k] = ngram[k+1];
      ngram[k] = sents[i][j];
      CalcDists(ngram, curr_ctxt_dist);
      // cerr << "ngram: " << ngram << " ||| dist: " << curr_ctxt_dist.second << endl;
      auto it = ctxt_map.find(curr_ctxt_dist);
      if(it != ctxt_map.end()) {
        cache_ids[i][j] = it->second;
      } else {
        cache_ids[i][j] = ctxt_map.size();
        ctxt_map.insert(make_pair(curr_ctxt_dist, ctxt_map.size()));
      }
    }
  }
  int num_dist_ctxt = num_dist_+num_ctxt_;
  cache_.resize(ctxt_map.size());
  for(auto it : ctxt_map)
    cache_[it.second] = it.first;
  LoadDists(0);
}

// Calculate training loss for one word
Expression SoftmaxMod::CalcLoss(Expression & in, const Sentence & ngram, bool train) {
  // Calculate contexts and distributions  
  CtxtDist ctxt_dist; ctxt_dist.first.resize(num_ctxt_); ctxt_dist.second.resize(num_dist_);
  CalcDists(ngram, ctxt_dist);
  return CalcLossExpr(in, ctxt_dist, *ngram.rbegin(), train);
}

// Calculate training loss for multiple words
Expression SoftmaxMod::CalcLoss(Expression & in, const vector<Sentence> & ngrams, bool train) {
  CtxtDist ctxt_dist; ctxt_dist.first.resize(num_ctxt_); ctxt_dist.second.resize(num_dist_);
  CtxtDist ctxt_dist_batch; ctxt_dist_batch.first.resize(num_ctxt_*ngrams.size()); ctxt_dist_batch.second.resize(num_dist_*ngrams.size());
  auto ctxt_it = ctxt_dist_batch.first.begin(); auto dist_it = ctxt_dist_batch.second.begin();
  vector<unsigned> words(ngrams.size());
  for(size_t i = 0; i < ngrams.size(); i++) {
    CalcDists(ngrams[i], ctxt_dist);
    for(float c : ctxt_dist.first) *ctxt_it++ = c;
    for(float d : ctxt_dist.second) *dist_it++ = d;
    words[i] = *ngrams[i].rbegin();
  }
  return CalcLossExpr(in, ctxt_dist_batch, words, train);
}

Expression SoftmaxMod::CalcLossCache(Expression & in, int cache_id, const Sentence & ngram, bool train) {
  return CalcLossExpr(in, cache_[cache_id], *ngram.rbegin(), train);
}

Expression SoftmaxMod::CalcLossCache(Expression & in, const vector<int> & cache_ids, const vector<Sentence> & ngrams, bool train) {
  assert(cache_ids.size() == ngrams.size());
  // Set up the context
  CtxtDist batched_cd;
  batched_cd.first.resize(num_ctxt_*cache_ids.size()); auto ctxt_it = batched_cd.first.begin();
  batched_cd.second.resize(num_dist_*cache_ids.size()); auto dist_it = batched_cd.second.begin();
  // Set up the ngrams
  vector<unsigned> words(ngrams.size());
  for(size_t i = 0; i < cache_ids.size(); i++) {
    for(float c : cache_[cache_ids[i]].first) *ctxt_it++ = c;
    for(float d : cache_[cache_ids[i]].second) *dist_it++ = d;
    words[i] = *ngrams[i].rbegin();
  }
  return CalcLossExpr(in, batched_cd, words, train);
}

Expression SoftmaxMod::CalcLossExpr(Expression & in, const CtxtDist & ctxt_dist, WordId wid, bool train) {
  // Create expressions
  Expression ctxt_expr = input(*in.pg, {(unsigned int)num_ctxt_}, ctxt_dist.first);
  Expression in_ctxt_expr = concatenate({in, ctxt_expr});
  Expression score_sms = affine_transform({i_sms_b_, i_sms_W_, in_ctxt_expr});
  // Do dropout and use only the regular softmax
  uniform_real_distribution<float> float_distribution(0.0, 1.0);
  if(train && (finished_words_ < drop_words_ || float_distribution(*cnn::rndeng) < dropout_)) {  
    finished_words_++;
    return pickneglogsoftmax(score_sms, wid);
  // Do no dropout
  } else {
    finished_words_++;
    Expression score_smd = affine_transform({i_smd_b_, i_smd_W_, in_ctxt_expr});
    Expression score = softmax(concatenate({score_sms, score_smd}));
    // Do mixture of distributions
    Expression word_prob = pick(score, wid) + input(*in.pg, {1, (unsigned int)num_dist_}, ctxt_dist.second) * pickrange(score, vocab_->size(), vocab_->size()+num_dist_);
    return -log(word_prob);
  }
}

Expression SoftmaxMod::CalcLossExpr(Expression & in, const CtxtDist & ctxt_dist_batched, const vector<unsigned> & wids, bool train) {
  // Create expressions
  Expression ctxt_expr = input(*in.pg, cnn::Dim({(unsigned int)num_ctxt_}, wids.size()), ctxt_dist_batched.first);
  Expression in_ctxt_expr = concatenate({in, ctxt_expr});
  Expression score_sms = affine_transform({i_sms_b_, i_sms_W_, in_ctxt_expr});
  // Do dropout and use only the regular softmax
  uniform_real_distribution<float> float_distribution(0.0, 1.0);
  if(train && (finished_words_ < drop_words_ || float_distribution(*cnn::rndeng) < dropout_)) {  
    finished_words_ += wids.size();
    return pickneglogsoftmax(score_sms, wids);
  // Do no dropout
  } else {
    finished_words_ += wids.size();
    Expression score_smd = affine_transform({i_smd_b_, i_smd_W_, in_ctxt_expr});
    Expression score = softmax(concatenate({score_sms, score_smd}));
    // Do mixture of distributions
    Expression word_prob = pick(score, wids) + input(*in.pg, cnn::Dim({1, (unsigned int)num_dist_}, wids.size()), ctxt_dist_batched.second) * pickrange(score, vocab_->size(), vocab_->size()+num_dist_);
    return -log(word_prob);
  }
}

// Calculate the full probability distribution
Expression SoftmaxMod::CalcProb(Expression & in, const Sentence & ctxt_ngram, bool train) {
  // Calculate the distributions
  CtxtDist ctxt_dist; ctxt_dist.first.resize(num_ctxt_); ctxt_dist.second.resize(num_dist_*vocab_->size());
  CalcAllDists(ctxt_ngram, ctxt_dist);
  // Create expressions
  Expression ctxt_expr = input(*in.pg, {(unsigned int)num_ctxt_}, ctxt_dist.first);
  Expression in_ctxt_expr = concatenate({in, ctxt_expr});
  Expression score_sms = affine_transform({i_sms_b_, i_sms_W_, in_ctxt_expr});
  Expression word_prob;
  uniform_real_distribution<float> float_distribution(0.0, 1.0);
  if(train && (finished_words_ < drop_words_ || float_distribution(*cnn::rndeng) < dropout_)) {  
    finished_words_++;
    word_prob = softmax(score_sms);
  } else {
    finished_words_++;
    Expression score_smd = affine_transform({i_smd_b_, i_smd_W_, in_ctxt_expr});
    Expression score = softmax(concatenate({score_sms, score_smd}));
    // Do mixture of distributions
    Expression dists = input(*in.pg, {(unsigned int)num_dist_, (unsigned int)vocab_->size()}, ctxt_dist.second);
    word_prob = pickrange(score, 0, vocab_->size()) + transpose(dists) * pickrange(score, vocab_->size(), vocab_->size()+num_dist_);
  }
  // cerr << "Word " << GlobalVars::curr_word << " and surrounding probs: " << as_vector(pickrange(word_prob, max(0,GlobalVars::curr_word-3), min(GlobalVars::curr_word+4, (int)vocab_->size())).value()) << endl;
  // DEBUG
  std::vector<float> prob_vec = as_vector(word_prob.value());
  float prob_val = 0;
  for(float f : prob_vec)
    prob_val += f;
  cerr << "prob_val = " << prob_val << endl;
  if(prob_val < 0 || prob_val > 1.001) {
    THROW_ERROR("Out of range probability: " << prob_val);
  }

  return word_prob;
}
Expression SoftmaxMod::CalcProb(Expression & in, const vector<Sentence> & ctxt_ngrams, bool train) {
  THROW_ERROR("SoftmaxMod::CalcProb Not implemented yet");
}
Expression SoftmaxMod::CalcLogProb(Expression & in, const Sentence & ctxt_ngram, bool train) {
  return log(CalcProb(in, ctxt_ngram, train));
}
Expression SoftmaxMod::CalcLogProb(Expression & in, const vector<Sentence> & ctxt_ngrams, bool train) {
  return log(CalcProb(in, ctxt_ngrams, train));
}

Expression SoftmaxMod::CalcProbCache(Expression & in, int cache_id,                  const Sentence & ctxt_ngram, bool train) {
  THROW_ERROR("SoftmaxMod::CalcProbCache Not implemented yet");
}
Expression SoftmaxMod::CalcProbCache(Expression & in, const Sentence & cache_ids, const vector<Sentence> & ctxt_ngrams, bool train) {
  THROW_ERROR("SoftmaxMod::CalcProbCache Not implemented yet");
}
Expression SoftmaxMod::CalcLogProbCache(Expression & in, int cache_id,                   const Sentence & ctxt_ngram, bool train) {
  THROW_ERROR("SoftmaxMod::CalcProbCache Not implemented yet");
}
Expression SoftmaxMod::CalcLogProbCache(Expression & in, const Sentence & cache_ids,  const vector<Sentence> & ctxt_ngrams, bool train) {
  THROW_ERROR("SoftmaxMod::CalcProbCache Not implemented yet");
}
