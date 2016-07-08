#include <lamtram/softmax-diff.h>
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

void SoftmaxDiff::LoadDists(int id) {
  if(dist_id_ == id) return;
  if(dist_files_.size() != 1) {
    assert(id < (int)dist_files_.size());
    cerr << "Loading distribution: " << dist_files_[id] << "..." << endl;
    dist_ptr_ = DistFactory::from_file(dist_files_[id], vocab_);
  } else if(dist_ptr_.get() != NULL) {
    cerr << "Loading distribution: " << dist_files_[0] << "..." << endl;
    dist_ptr_ = DistFactory::from_file(dist_files_[0], vocab_);
  }
  if(dist_ptr_->get_dense_size() != 1 || dist_ptr_->get_sparse_size() > 0)
    THROW_ERROR("Distribution must have exactly one dense distribution");
  dist_id_ = id;
}

SoftmaxDiff::SoftmaxDiff(const string & sig, int input_size, const DictPtr & vocab, cnn::Model & mod)
                        : SoftmaxBase(sig,input_size,vocab,mod), vocab_size_(vocab->size()), 
                          finished_words_(0), drop_words_(0), dropout_(0.f), dist_id_(-1) {
  vector<string> strs = Tokenize(sig, ":");
  if(strs.size() <= 2 || strs[0] != "diff") THROW_ERROR("Bad signature in SoftmaxDiff: " << sig);
  string dist_file;
  // Read the arguments
  for(size_t i = 1; i < strs.size(); i++) {
    if(strs[i].substr(0, 8) == "dropout=") {
      dropout_ = stof(strs[i].substr(8));
    } else if(strs[i].substr(0, 10) == "dropwords=") {
      drop_words_ = stof(strs[i].substr(10));
    } else if(strs[i].substr(0, 5) == "dist=") {
      if(dist_file != "") THROW_ERROR("Can only have one distribution");
      dist_file = strs[i].substr(5);
    } else if(strs[i].substr(0, 10) == "wildcards=") {
      wildcards_ = Tokenize(strs[i].substr(10), "|");
    } else {
      THROW_ERROR("Illegal option in SoftmaxDiff initializer: " << strs[i]);
    }
  }
  // Read in the distributions
  if(dist_file == "") THROW_ERROR("must specify a distribution");
  dist_files_ = TokenizeWildcarded(dist_file,wildcards_,"|");
  // Read the distributions for the first set of training data
  LoadDists(0);
  ctxt_len_ = dist_ptr_->get_ctxt_len();
  // Initialize the parameters
  p_sm_W_ = mod.add_parameters({(unsigned int)vocab_size_, (unsigned int)input_size});
  p_sm_b_ = mod.add_parameters({(unsigned int)vocab_size_});  
}

void SoftmaxDiff::NewGraph(cnn::ComputationGraph & cg) {
  i_sm_b_ = parameter(cg, p_sm_b_);
  i_sm_W_ = parameter(cg, p_sm_W_);
}

// Calculate the context and distribution for the current ngram
void SoftmaxDiff::CalcAllDists(const Sentence & ctxt_ngram, std::vector<float> & dist) {
  vector<pair<pair<int,int>,float> > sparse_dist;
  int dense_offset = 0, sparse_offset = 0;
  dist_ptr_->calc_all_word_dists(ctxt_ngram, vocab_size_, 1.f/vocab_size_, 1.f, dist, dense_offset, sparse_dist, sparse_offset);
  for(auto & val : dist) val = log(val);
}

void SoftmaxDiff::Cache(const vector<Sentence> & sents, const vector<int> & set_ids, vector<Sentence> & cache_ids) {
  assert(sents.size() == set_ids.size());
  // Create pairs of context floats, distribution floats
  std::vector<float> curr_ctxt_dist(vocab_size_);
  std::unordered_map<std::vector<float>, int> ctxt_map;
  std::unordered_map<pair<int,Sentence>, int> ctxt_ngram_map;
  size_t i, j, k;
  // Fill the cache with values
  cache_ids.resize(sents.size());
  for(i = 0; i < sents.size(); i++) {
    LoadDists(set_ids[i]+1);
    Sentence ctxt_ngram(ctxt_len_, 0);
    cache_ids[i].resize(sents[i].size());
    for(j = 0; j < sents[i].size(); j++) {
      pair<int,Sentence> idngram(set_ids[i],ctxt_ngram);
      auto itngram = ctxt_ngram_map.find(idngram);
      if(itngram != ctxt_ngram_map.end()) {
        cache_ids[i][j] = itngram->second;
      } else {
        CalcAllDists(ctxt_ngram, curr_ctxt_dist);
        // cerr << "ngram: " << ngram << " ||| dist: " << curr_ctxt_dist.second << endl;
        auto it = ctxt_map.find(curr_ctxt_dist);
        if(it != ctxt_map.end()) {
          cache_ids[i][j] = it->second;
        } else {
          cache_ids[i][j] = ctxt_map.size();
          ctxt_map.insert(make_pair(curr_ctxt_dist, cache_ids[i][j]));
          ctxt_ngram_map.insert(make_pair(idngram, cache_ids[i][j]));
        }
      }
      if(ctxt_ngram.size()) {
        for(k = 0; k < ctxt_ngram.size()-1; k++)
          ctxt_ngram[k] = ctxt_ngram[k+1];
        ctxt_ngram[k] = sents[i][j];
      }
    }
  }
  cache_.resize(ctxt_map.size());
  for(auto it : ctxt_map)
    cache_[it.second] = it.first;
  LoadDists(0);
}

// Calculate training loss for one word
Expression SoftmaxDiff::CalcLoss(Expression & in, Expression & prior, const Sentence & ngram, bool train) {
  assert(prior.pg == nullptr);
  // Calculate contexts and distributions  
  std::vector<float> ctxt_dist(vocab_size_);
  Sentence ctxt_ngram(ngram); ctxt_ngram.resize(ngram.size()-1);
  CalcAllDists(ctxt_ngram, ctxt_dist);
  return CalcLossExpr(in, prior, ctxt_dist, *ngram.rbegin(), train);
}

// Calculate training loss for multiple words
Expression SoftmaxDiff::CalcLoss(Expression & in, Expression & prior, const vector<Sentence> & ngrams, bool train) {
  assert(prior.pg == nullptr);
  std::vector<float> ctxt_dist(vocab_size_);
  std::vector<float> ctxt_dist_batch(vocab_size_*ngrams.size());
  vector<unsigned> words(ngrams.size());
  for(size_t i = 0; i < ngrams.size(); i++) {
    Sentence ctxt_ngram(ngrams[i]); ctxt_ngram.resize(ngrams[i].size()-1);
    // cerr << "CalcLoss @ " << i << ": ctxt_ngram=" << ctxt_ngram << endl;
    CalcAllDists(ctxt_ngram, ctxt_dist);
    memcpy(&ctxt_dist_batch[i*vocab_size_], &ctxt_dist[0], vocab_size_*sizeof(float));
    words[i] = *ngrams[i].rbegin();
  }
  return CalcLossExpr(in, prior, ctxt_dist_batch, words, train);
}

Expression SoftmaxDiff::CalcLossCache(Expression & in, Expression & prior, int cache_id, const Sentence & ngram, bool train) {
  return CalcLossExpr(in, prior, cache_[cache_id], *ngram.rbegin(), train);
}

Expression SoftmaxDiff::CalcLossCache(Expression & in, Expression & prior, const vector<int> & cache_ids, const vector<Sentence> & ngrams, bool train) {
  assert(prior.pg == nullptr);
  assert(cache_ids.size() == ngrams.size());
  // Set up the context
  std::vector<float> batched_cd(vocab_size_*cache_ids.size());
  // Set up the ngrams
  vector<unsigned> words(ngrams.size());
  for(size_t i = 0; i < cache_ids.size(); i++) {
    memcpy(&batched_cd[i*vocab_size_], &cache_[cache_ids[i]][0], vocab_size_*sizeof(float));
    words[i] = *ngrams[i].rbegin();
  }
  return CalcLossExpr(in, prior, batched_cd, words, train);
}

Expression SoftmaxDiff::CalcLossExpr(Expression & in, Expression & prior, const std::vector<float> & ctxt_dist, WordId wid, bool train) {
  assert(prior.pg == nullptr);
  // Create expressions
  Expression score = affine_transform({i_sm_b_, i_sm_W_, in});
  uniform_real_distribution<float> float_distribution(0.0, 1.0);
  if(!(train && (finished_words_ < drop_words_ || float_distribution(*cnn::rndeng) < dropout_))) {
    // cerr << "score before=" << as_scalar(pick(score, wid).value());
    // cerr << "ctxt_dist[" << wid << "] == " << ctxt_dist[wid] << endl;
    score = score + input(*in.pg, {(unsigned int)vocab_size_}, ctxt_dist);
    // cerr << " after=" << as_scalar(pick(score, wid).value()) << endl;
  }
  finished_words_++;
  return pickneglogsoftmax(score, wid);
}

Expression SoftmaxDiff::CalcLossExpr(Expression & in, Expression & prior, const std::vector<float> & ctxt_dist_batched, const vector<unsigned> & wids, bool train) {
  assert(prior.pg == nullptr);
  Expression score = affine_transform({i_sm_b_, i_sm_W_, in});
  uniform_real_distribution<float> float_distribution(0.0, 1.0);
  if(!(train && (finished_words_ < drop_words_ || float_distribution(*cnn::rndeng) < dropout_))) {
    // cerr << "score before=" << as_vector(pick(score, wids).value());
    score = score + input(*in.pg, cnn::Dim({(unsigned int)vocab_size_}, wids.size()), ctxt_dist_batched);
    // cerr << " after=" << as_vector(pick(score, wids).value()) << endl;
  }
  finished_words_ += wids.size();
  return pickneglogsoftmax(score, wids);
}

// Calculate the full probability distribution
Expression SoftmaxDiff::CalcProb(Expression & in, Expression & prior, const Sentence & ctxt_ngram, bool train) {
  assert(prior.pg == nullptr);
  // Calculate the distributions
  std::vector<float> ctxt_dist(vocab_size_);
  CalcAllDists(ctxt_ngram, ctxt_dist);
  // Create expressions
  Expression score = affine_transform({i_sm_b_, i_sm_W_, in});
  uniform_real_distribution<float> float_distribution(0.0, 1.0);
  if(!(train && (finished_words_ < drop_words_ || float_distribution(*cnn::rndeng) < dropout_))) {
    score = score + input(*in.pg, {(unsigned int)vocab_size_}, ctxt_dist);
  }
  finished_words_++;
  return softmax(score);
}
Expression SoftmaxDiff::CalcProb(Expression & in, Expression & prior, const vector<Sentence> & ctxt_ngrams, bool train) {
  THROW_ERROR("SoftmaxDiff::CalcProb Not implemented yet");
}
Expression SoftmaxDiff::CalcLogProb(Expression & in, Expression & prior, const Sentence & ctxt_ngram, bool train) {
  return log(CalcProb(in, prior, ctxt_ngram, train));
}
Expression SoftmaxDiff::CalcLogProb(Expression & in, Expression & prior, const vector<Sentence> & ctxt_ngrams, bool train) {
  return log(CalcProb(in, prior, ctxt_ngrams, train));
}

Expression SoftmaxDiff::CalcProbCache(Expression & in, Expression & prior, int cache_id,                  const Sentence & ctxt_ngram, bool train) {
  THROW_ERROR("SoftmaxDiff::CalcProbCache Not implemented yet");
}
Expression SoftmaxDiff::CalcProbCache(Expression & in, Expression & prior, const Sentence & cache_ids, const vector<Sentence> & ctxt_ngrams, bool train) {
  THROW_ERROR("SoftmaxDiff::CalcProbCache Not implemented yet");
}
Expression SoftmaxDiff::CalcLogProbCache(Expression & in, Expression & prior, int cache_id,                   const Sentence & ctxt_ngram, bool train) {
  THROW_ERROR("SoftmaxDiff::CalcProbCache Not implemented yet");
}
Expression SoftmaxDiff::CalcLogProbCache(Expression & in, Expression & prior, const Sentence & cache_ids,  const vector<Sentence> & ctxt_ngrams, bool train) {
  THROW_ERROR("SoftmaxDiff::CalcProbCache Not implemented yet");
}
