#include <boost/algorithm/string.hpp>
#include <cnn/dict.h>
#include <lamtram/macros.h>
#include <lamtram/sentence.h>
#include <lamtram/counts.h>
#include <lamtram/dict-utils.h>

using namespace lamtram;
using namespace std;

void Counts::add_count(const Sentence & idx, WordId wid, WordId last_fallback) {
  for(auto id : idx)
    if(id == -1) THROW_ERROR("Illegal context");
  ContextCountsPtr & my_cnts = cnts_[idx];
  if(my_cnts.get() == NULL) my_cnts.reset(new_counts_ptr());
  my_cnts->full_sum++;
  my_cnts->cnts[wid]++;
}

// Calculate 4 contextual features (1-3 are logs)
// 1) The total sum of all counts for this context
// 2) The number of unique following words in this context
// 3) The sum of 1 and 2
// 4) A binary feature indicating unseen contexts
void Counts::calc_ctxt_feats(const Sentence & ctxt, float * fl) {
  auto it = cnts_.find(ctxt);
  if(it == cnts_.end()) {
    fl[0] = 0.0;
    fl[1] = 0.0;
    fl[2] = 1.0;
  } else {
    assert(it->second.get() != NULL);
    float full_sum = it->second->full_sum;
    float uniq = it->second->cnts.size();
    fl[0] = log(full_sum);
    fl[1] = log(uniq);
    fl[2] = 0.0;
  }
}

// Calculate 6 contextual features (1-3 and 5-6 are logs)
// 1) The total sum of all counts for this context
// 2) The number of unique following words in this context
// 3) The sum of 1 and 2
// 4) A binary feature indicating unseen contexts
// 5) The discounted sum of all counts for this context
// 6) The difference between the total sum and discounted sum
void CountsMabs::calc_ctxt_feats(const Sentence & ctxt, float * fl) {
  auto it = cnts_.find(ctxt);
  if(it == cnts_.end()) {
    fl[0] = 0.0;
    fl[1] = 0.0;
    fl[2] = 0.0;
    fl[3] = 1.0;
  } else {
    assert(it->second.get() != NULL);
    float full_sum = it->second->full_sum;
    float uniq = it->second->cnts.size();
    float disc_sum = ((ContextCountsDisc*)it->second.get())->disc_sum;
    fl[0] = log(full_sum);
    fl[1] = log(uniq);
    fl[2] = log(disc_sum);
    fl[3] = 0.0;
  }
}

// Calculate the ctxtual features 
void Counts::calc_word_dists(const Sentence & ngram,
                             float uniform_prob,
                             float unk_prob,
                             DistTarget & trg,
                             int & dense_offset) const {
  Sentence ctxt = ngram; ctxt.resize(ctxt.size() - 1);
  WordId wid = *ngram.rbegin();
  auto it = cnts_.find(ctxt);
  if(it == cnts_.end()) {
    trg.first[dense_offset] = (wid == 0 ? unk_prob * uniform_prob : uniform_prob);
  } else {
    auto it2 = it->second->cnts.find(wid);
    trg.first[dense_offset] = (it2 != it->second->cnts.end() ?
                               mod_cnt(it2->second) / it->second->get_denominator() :
                               0.0);
    if(wid == 0) trg.first[dense_offset] *= unk_prob;
  }
  dense_offset++;
}

void Counts::write(DictPtr dict, std::ostream & out) const {
  for(const auto & cnt : cnts_) {
    out << PrintWords(*dict, cnt.first) << '\t';
    std::string prev = "";
    assert(cnt.second.get() != NULL);
    for(const auto & kv : cnt.second->cnts) {
      out << prev << dict->Convert(kv.first) << ' ' << kv.second; 
      prev = " ";
    }
    out << std::endl;
  }
  out << std::endl;
}

void Counts::read(DictPtr dict, std::istream & in) {
  std::string line;
  std::vector<std::string> strs, words;
  while(getline(in, line)) {
    if(line == "") break;
    boost::split(strs, line, boost::is_any_of("\t"));
    assert(strs.size() == 2);
    ContextCountsPtr & ptr = cnts_[ParseWords(*dict, strs[0], false)];
    assert(ptr.get() == NULL);
    ptr.reset(new_counts_ptr());
    boost::split(words, strs[1], boost::is_any_of(" "));
    assert(words.size() % 2 == 0);
    for(size_t i = 0; i < words.size(); i += 2) {
      int cnt = stoi(words[i+1]);
      ptr->add_word(dict->Convert(words[i]), cnt, mod_cnt(cnt));
    }
  }
}

void CountsMabs::finalize_count() {
  discounts_.resize(4);
  std::vector<int> fofs(5);
  for(auto & kv : cnts_)
    for(auto & cnt : kv.second->cnts)
      if(cnt.second < fofs.size())
        fofs[cnt.second]++;
  float Y = fofs[1] / float(fofs[1] + 2*fofs[2]);
  discounts_[1] = 1 - 2.0*Y*fofs[2]/fofs[1];
  discounts_[2] = 2 - 3.0*Y*fofs[3]/fofs[2];
  discounts_[3] = 3 - 4.0*Y*fofs[4]/fofs[3];
  for(auto & kv : cnts_) {
    ContextCountsDisc* count = (ContextCountsDisc*)kv.second.get();
    count->disc_sum = count->full_sum;
    for(auto & cnt : count->cnts)
      count->disc_sum -= (cnt.second - mod_cnt(cnt.second));
  }
}

float CountsMabs::mod_cnt(int cnt) const {
  return cnt > 0 ? cnt - discounts_[std::min(cnt,(int)discounts_.size()-1)] : 0.0;
}

void CountsMabs::write(DictPtr dict, std::ostream & out) const {
  out << discounts_[0];
  for(size_t i = 1; i < discounts_.size(); i++)
    out << ' ' << discounts_[i];
  out << std::endl;
  Counts::write(dict, out);
}

void CountsMabs::read(DictPtr dict, std::istream & in) {
  std::string line;
  if(!getline(in, line))
    THROW_ERROR("Premature end");
  std::vector<std::string> strs;
  boost::split(strs, line, boost::is_any_of(" "));
  for(const std::string & f : strs)
    discounts_.push_back(stof(f));
  Counts::read(dict, in);
}

void CountsMkn::add_count(const Sentence & idx, WordId wid, WordId last_fallback) {
  if(last_fallback == -1) THROW_ERROR("Passing ctxt with no last word to kneser ney");
  ContextCountsUniqPtr & my_cnts = cnts_uniq_[idx];
  if(my_cnts.get() == NULL) my_cnts.reset(new ContextCountsUniq);
  auto & my_set = my_cnts->second[wid];
  if(my_set.find(last_fallback) == my_set.end()) {
    my_cnts->first++;
    my_set.insert(last_fallback);
  }
}

void CountsMkn::finalize_count() {
  for(auto & kv_uniq : cnts_uniq_) {
    auto & kv = cnts_[kv_uniq.first];
    kv.reset(new ContextCountsDisc);
    assert(kv_uniq.second != NULL);
    for(auto & cnt : kv_uniq.second->second)
      kv->cnts[cnt.first] = cnt.second.size();
  }
  cnts_uniq_.clear();
  CountsMabs::finalize_count();
}
