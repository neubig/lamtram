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

SoftmaxMod::SoftmaxMod(const string & sig, int input_size, const DictPtr & vocab, cnn::Model & mod) : SoftmaxBase(sig,input_size,vocab,mod), num_dist_(0), num_ctxt_(0), dist_id_(-1) {
  vector<string> strs = Tokenize(sig, ":");
  if(strs.size() <= 2 || strs[0] != "mod") THROW_ERROR("Bad signature in SoftmaxMod: " << sig);
  vector<string> my_dist_files;
  // Read the arguments
  for(size_t i = 1; i < strs.size(); i++) {
    if(strs[i].substr(0, 8) == "dropout=") {
      dropout_ = stof(strs[i].substr(8));
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
  LoadDists(1);
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

void SoftmaxMod::Cache(const vector<Sentence> sents, const vector<int> set_ids) {
  assert(sents.size() == set_ids.size());
  // Create pairs of context floats, distribution floats
  CtxtDist curr_ctxt_dist; curr_ctxt_dist.first.resize(num_ctxt_); curr_ctxt_dist.second.resize(num_dist_);
  std::unordered_map<CtxtDist, int> ctxt_map;
  size_t i, j, k;
  // Fill the cache with values
  cache_ids_.resize(sents.size());
  for(i = 0; i < sents.size(); i++) {
    LoadDists(set_ids[i]+1);
    Sentence ngram(ctxt_len_+1, 0);
    cache_ids_[i].resize(sents[i].size());
    for(j = 0; j < sents[i].size(); j++) {
      for(k = 0; k < ngram.size()-1; k++)
        ngram[k] = ngram[k+1];
      ngram[k] = sents[i][j];
      CalcDists(ngram, curr_ctxt_dist);
      // cerr << "ngram: " << ngram << " ||| dist: " << curr_ctxt_dist.second << endl;
      auto it = ctxt_map.find(curr_ctxt_dist);
      if(it != ctxt_map.end()) {
        cache_ids_[i][j] = it->second;
      } else {
        cache_ids_[i][j] = ctxt_map.size();
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

Expression SoftmaxMod::CalcLossCache(Expression & in, const Sentence & ngram, pair<int,int> sent_word, bool train) {
  assert(cache_ids_.size() > sent_word.first);
  assert(cache_ids_[sent_word.first].size() > sent_word.second);
  assert(cache_.size() > cache_ids_[sent_word.first][sent_word.second]);
  return CalcLossExpr(in, cache_[cache_ids_[sent_word.first][sent_word.second]], *ngram.rbegin(), train);
}

Expression SoftmaxMod::CalcLossCache(Expression & in, const vector<Sentence> & ngram, const vector<pair<int,int> > & sent_words, bool train) {
  THROW_ERROR("SoftmaxMod::CalcLossCache Not implemented yet");
}

Expression SoftmaxMod::CalcLossExpr(Expression & in, const CtxtDist & ctxt_dist, WordId wid, bool train) {
  // Create expressions
  Expression ctxt_expr = input(*in.pg, {(unsigned int)num_ctxt_}, ctxt_dist.first);
  Expression in_ctxt_expr = concatenate({in, ctxt_expr});
  Expression score_sms = affine_transform({i_sms_b_, i_sms_W_, in_ctxt_expr});
  // Do dropout and use only the regular softmax
  uniform_real_distribution<float> float_distribution(0.0, 1.0);
  if(train && float_distribution(*cnn::rndeng) < dropout_) {  
    return pickneglogsoftmax(score_sms, wid);
  // Do no dropout
  } else {
    Expression score_smd = affine_transform({i_smd_b_, i_smd_W_, in_ctxt_expr});
    Expression score = softmax(concatenate({score_sms, score_smd}));
    // Do mixture of distributions
    Expression word_prob = pick(score, wid) + input(*in.pg, {1, (unsigned int)num_dist_}, ctxt_dist.second) * pickrange(score, vocab_->size(), vocab_->size()+num_dist_);
    return -log(word_prob);
  }
}

// Calculate training loss for multiple words
Expression SoftmaxMod::CalcLoss(Expression & in, const vector<Sentence> & ngrams, bool train) {
  THROW_ERROR("SoftmaxMod::CalcLoss Not implemented yet");
}

// Calculate the full probability distribution
Expression SoftmaxMod::CalcProbability(Expression & in) {
  THROW_ERROR("SoftmaxMod::CalcLogProbability Not implemented yet");
}
Expression SoftmaxMod::CalcLogProbability(Expression & in) {
  THROW_ERROR("SoftmaxMod::CalcLogProbability Not implemented yet");
}

