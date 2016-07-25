
#include <boost/range/algorithm/max_element.hpp>
#include <boost/range/algorithm/min_element.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/irange.hpp>
#include <cnn/dict.h>
#include <lamtram/macros.h>
#include <lamtram/dist-one-hot.h>

using namespace std;
using namespace lamtram;

// Signature should be of the form
// 1) onehot
DistOneHot::DistOneHot(const std::string & sig) : DistBase(sig) {
  if(sig != "onehot")
    THROW_ERROR("Bad signature in DistOneHot: " << sig);
}

std::string DistOneHot::get_sig() const {
  return "onehot";
}

// Add stats from one sentence at training time for count-based models
void DistOneHot::add_stats(const Sentence & sent) {
  for(auto w : sent) {
    if(w == -1) continue;
    auto it = mapping_.find(w);
    if(it == mapping_.end()) {
      mapping_[w] = back_mapping_.size();
      back_mapping_.push_back(w);
    }
  }
}

void DistOneHot::finalize_stats() {
}

// Get the number of ctxtual features we can expect from this model
size_t DistOneHot::get_ctxt_size() const {
  return 0;
}

// And calculate these features
void DistOneHot::calc_ctxt_feats(const Sentence & ctxt, float* feats_out) const {
}

// Calculate the probability of the last word in ngram given the previous words.
// uniform_prob and unk_prob are the uniform probability and penalty given to unknown
// words. trg is the output, and dense_offset and sparse_offset offset the dense
// and sparse distributions respectively.
void DistOneHot::calc_word_dists(const Sentence & ngram,
                                float uniform_prob,
                                float unk_prob,
                                std::vector<float> & trg_dense,
                                int & dense_offset,
                                std::vector<std::pair<int,float> > & trg_sparse,
                                int & sparse_offset) const {
  WordId wid = *ngram.rbegin();
  auto it = mapping_.find(wid);
  if(it != mapping_.end())
    trg_sparse.push_back(make_pair(sparse_offset+it->second, (wid == 0 ? unk_prob : 1.0)));
  sparse_offset += back_mapping_.size();
}

// Read/write model. If dict is null, use numerical ids, otherwise strings.
#define DIST_ONEHOT_VERSION "distonehot_v1"
void DistOneHot::write(DictPtr dict, std::ostream & out) const {
  out << DIST_ONEHOT_VERSION << endl;
  for(auto i : back_mapping_)
    out << dict->convert(i) << endl;
  out << endl;
}
void DistOneHot::read(DictPtr dict, std::istream & in) {
  string line;
  if(!(getline(in, line) && line == DIST_ONEHOT_VERSION))
    THROW_ERROR("Bad format in DistOneHot");
  while(getline(in, line)) {
    if(line == "") break;
    WordId id = dict->convert(line);
    if(id == -1) THROW_ERROR("Out-of-vocabulary word found in one hot model: " << line);
    mapping_[id] = back_mapping_.size();
    back_mapping_.push_back(id);
  }
}
