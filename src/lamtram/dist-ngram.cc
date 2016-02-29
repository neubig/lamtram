
#include <boost/range/algorithm/max_element.hpp>
#include <boost/range/algorithm/min_element.hpp>
#include <boost/algorithm/string.hpp>
#include <cnn/dict.h>
#include <lamtram/macros.h>
#include <lamtram/dist-ngram.h>
#include <lamtram/dict-utils.h>
#include <lamtram/counts.h>

#define EOS_ID 0
#define UNK_ID 1

using namespace std;
using namespace lamtram;

// Signature should be of the form
// 1) ngram
// 2) lin/mabs/mkn: where "lin" means linear, "mabs" means modified absolute
//    discounting, and "mkn" means modified kneser ney
DistNgram::DistNgram(const std::string & sig) : DistBase(sig) {
  // Split and sanity check signature
  std::vector<std::string> strs;
  boost::split(strs,sig,boost::is_any_of("_"));
  if((strs[0] != "ngram" && strs[0] != "ngramh") || strs.size() < 2)
    THROW_ERROR("Bad signature in DistNgram: " << sig);
  heuristics_ = (strs[0] == "ngramh");
  // Get the rest of the ctxt
  if(strs.size() > 1) {
    for(size_t i = 2; i < strs.size(); i++)
      ctxt_pos_.push_back(stoi(strs[i]));
    if(ctxt_pos_.size() != 0) {
      ctxt_len_ = *boost::max_element(ctxt_pos_);
      if(*boost::min_element(ctxt_pos_) < 1)
        THROW_ERROR("Negative ctxt id in signature: " << sig); 
    }
  }
  ngram_len_ = ctxt_pos_.size() + 1;
  // save the smoothing type
  if(strs[1] == "lin") {
    smoothing_ = SMOOTH_LIN;
  } else if (strs[1] == "mabs") {
    smoothing_ = SMOOTH_MABS;
  } else if (strs[1] == "mkn") {
    smoothing_ = SMOOTH_MKN;
  } else {
    THROW_ERROR("Bad smoothing type in signature: " << sig);
  }
}

std::string DistNgram::get_sig() const {
  ostringstream oss;
  oss << "ngram" << (heuristics_?"h":"") << "_" << (smoothing_ == SMOOTH_LIN ? "lin" : "mabs");
  for(auto i : ctxt_pos_) oss << '_' << i;
  return oss.str();
}

inline WordId get_sym(const Sentence & sent, int pos) {
  return pos >= 0 ? sent[pos] : 0;
}

int DistNgram::get_ctxt_id(const Sentence & ngram) {
  auto it = mapping_.find(ngram);
  if(it != mapping_.end()) {
    return it->second;
  } else {
    int ret = ctxt_cnts_.size();
    ctxt_cnts_.push_back(DistNgramCounts());
    if(smoothing_ == SMOOTH_MKN) aux_cnts_.push_back(0);
    mapping_.insert(it, make_pair(ngram, ret));
    return ret;
  }
}
int DistNgram::get_tmp_ctxt_id(const Sentence & ngram) {
  auto it = mapping_.find(ngram);
  if(it != mapping_.end()) return it->second;
  auto it2 = tmp_mapping_.find(ngram);
  if(it2 != tmp_mapping_.end()) return it2->second;
  int ret = ctxt_cnts_.size();
  ctxt_cnts_.push_back(DistNgramCounts());
  if(smoothing_ == SMOOTH_MKN) aux_cnts_.push_back(0);
  tmp_mapping_.insert(it, make_pair(ngram, ret));
  return ret;
}
int DistNgram::get_existing_ctxt_id(const Sentence & ngram) const {
  auto it = mapping_.find(ngram);
  return it != mapping_.end() ? it->second : -1;
}

// Add stats from one sentence at training time for count-based models
void DistNgram::add_stats(const Sentence & sent) {  
  for(size_t i = 0; i < sent.size(); i++) {
    Sentence ngram(1, sent[i]);
    // for each context, add if necessary
    for(int j = ctxt_pos_.size(); j >= 0; j--) {
      if(j == 0) {
        mapping_[ngram]++;
      } else { 
        if(smoothing_ != SMOOTH_MKN)
          ctxt_cnts_[get_ctxt_id(ngram)].first++;
        else if(j != ctxt_pos_.size())
          aux_cnts_[get_ctxt_id(ngram)]++;
        ngram.insert(ngram.begin(), get_sym(sent, i-ctxt_pos_[j-1]));
      }
    }
  }
}

inline Sentence get_prev_ctxt(Sentence sent) {
  assert(sent.size() > 0);
  sent.resize(sent.size()-1);
  return sent;
}
inline Sentence get_next_ctxt(Sentence sent) {
  assert(sent.size() > 0);
  sent.erase(sent.begin());
  return sent;
}

void DistNgram::finalize_stats() {
  int here = 0;
  // Add the counts to the denominator 
  if(smoothing_ != SMOOTH_MKN || ngram_len_ == 1) {
    for(const auto & ngram : mapping_) {
      int value = (ngram.first.size() == ngram_len_ ? ngram.second : ctxt_cnts_[ngram.second].first);
      int id = get_tmp_ctxt_id(get_prev_ctxt(ngram.first));
      ctxt_cnts_[id].second += value;
      ctxt_cnts_[id].third++;
    }
  } else {
    for(const auto & ngram : mapping_) {
      if(ngram.first.size() == ngram_len_) {
        int id = get_tmp_ctxt_id(get_prev_ctxt(ngram.first));
        ctxt_cnts_[id].second += ngram.second;
        ctxt_cnts_[id].third++;
      }
      if(ngram.first.size() == ngram_len_ || aux_cnts_[ngram.second]) {  
        Sentence next_ctxt = get_next_ctxt(ngram.first);
        int val = ++ctxt_cnts_[get_tmp_ctxt_id(next_ctxt)].first;
        int prev_next = get_tmp_ctxt_id(get_prev_ctxt(next_ctxt));
        ++ctxt_cnts_[prev_next].second;
        if(val == 1) ++ctxt_cnts_[prev_next].third;
      }
    }
  }
  // Restore the temporary contexts
  for(const auto & tmp : tmp_mapping_)
    mapping_[tmp.first] = tmp.second;
  tmp_mapping_.clear();
  // Get scores for discounting if necessary
  if(smoothing_ == SMOOTH_MABS || smoothing_ == SMOOTH_MKN) {
    vector<vector<float> > fofs(ngram_len_, vector<float>(5));
    for(const auto & ngram : mapping_) {
      int value = (ngram.first.size() == ngram_len_ ? ngram.second : ctxt_cnts_[ngram.second].first);
      if(value > 0 && value <= 4)
        fofs[ngram.first.size()-1][value]++;
    }
    discounts_ = vector<vector<float> >(ngram_len_, vector<float>(4));
    for(size_t i = 0; i < fofs.size(); i++) {    
      float Y = fofs[i][1] / float(fofs[i][1] + 2*fofs[i][2]);
      discounts_[i][1] = 1 - 2.0*Y*fofs[i][2]/fofs[i][1];
      discounts_[i][2] = 2 - 3.0*Y*fofs[i][3]/fofs[i][2];
      discounts_[i][3] = 3 - 4.0*Y*fofs[i][4]/fofs[i][3];
      for(size_t j = 1; j < 4; j++) {
        if(discounts_[i][j] < 0) {
          cerr << "WARNING: negative discount=" << discounts_[i][j] << ". Setting to 0.5." << endl;
          discounts_[i][j] = 0.5;
        }
      }
    }
    // Perform discounting
    disc_ctxt_cnts_.resize(ctxt_cnts_.size());
    for(const auto & ngram : mapping_) {
      int value;
      if(ngram.first.size() == ngram_len_) {
        value = ngram.second;
      } else {
        value = ctxt_cnts_[ngram.second].first;
        disc_ctxt_cnts_[ngram.second] += ctxt_cnts_[ngram.second].second;
      }
      if(value > 0) {
        int id = get_existing_ctxt_id(get_prev_ctxt(ngram.first));
        assert(id != -1);
        disc_ctxt_cnts_[id] -= discounts_[ngram.first.size()-1][min(3,value)];
      }
    }
  }
}

// Get the number of ctxtual features we can expect from this model
size_t DistNgram::get_ctxt_size() const {
  return ngram_len_ * (smoothing_ == SMOOTH_LIN ? 3 : 4);
}

// And calculate these features
void DistNgram::calc_ctxt_feats(const Sentence & ctxt, float* feats_out) const {
  assert(ctxt.size() >= ctxt_len_);
  Sentence ngram;
  int ctxt_size = (smoothing_ == SMOOTH_LIN ? 3 : 4);
  for(int j = ctxt_pos_.size(); j >= 0; j--) {
    int id = get_existing_ctxt_id(ngram);
    if(id == -1 || ctxt_cnts_[id].second == 0) {
      for(int j2 = j; j2 >= 0; j2--) {
        *(feats_out++) = 1.f;
        for(size_t i = 1; i < ctxt_size; i++) *(feats_out++) = 0.f;
      }
      break;
    } else {
      *(feats_out++) = 0.f;
      *(feats_out++) = log(ctxt_cnts_[id].second);
      *(feats_out++) = log(ctxt_cnts_[id].third);
      if(smoothing_ != SMOOTH_LIN)
        *(feats_out++) = log(disc_ctxt_cnts_[id]);
    }
    if(j != 0)
      ngram.insert(ngram.begin(), ctxt[ctxt.size()-ctxt_pos_[j-1]]);
  }
}

void DistNgram::calc_word_dists(const Sentence & ngram,
                                float uniform_prob,
                                float unk_prob,
                                std::vector<float> & trg_dense,
                                int & dense_offset,
                                std::vector<std::pair<int,float> > & trg_sparse,
                                int & sparse_offset) const {
  // If heuristics, write to a temporary vector then adjust
  vector<float> temp_vec(heuristics_?ctxt_pos_.size()+1:0);
  int temp_offset = 0;
  vector<float> & write_vec = (heuristics_?temp_vec:trg_dense);
  int & write_offset = (heuristics_?temp_offset:dense_offset);
  // Calculate the probability
  float base_prob = (*ngram.rbegin() != UNK_ID ? 1.0 : unk_prob);
  Sentence this_ngram(1, *ngram.rbegin()), this_ctxt;
  for(int j = ctxt_pos_.size(); j >= 0; j--) {
    auto ngram_it = mapping_.find(this_ngram);
    auto context_it = mapping_.find(this_ctxt);
    if(ngram_it == mapping_.end()) {
      float my_prob = (context_it != mapping_.end() ? 0.0 : uniform_prob * base_prob);
      // cerr << "my_prob: " << my_prob << endl;
      write_vec[write_offset++] = my_prob;
    } else {
      assert(context_it != mapping_.end());
      int value = (j == 0 ? ngram_it->second : ctxt_cnts_[ngram_it->second].first);
      if(smoothing_ == SMOOTH_MABS || smoothing_ == SMOOTH_MKN) {
        // cerr << "value_absmkn: " << (value-discounts_[this_ctxt.size()][min(value,3)]) << "/" << disc_ctxt_cnts_[context_it->second] * base_prob << endl;
        write_vec[write_offset++] = (value-discounts_[this_ctxt.size()][min(value,3)])/disc_ctxt_cnts_[context_it->second] * base_prob;
      } else {
        // cerr << "value_lin: " << value << "/" << (float)ctxt_cnts_[context_it->second].second * base_prob << endl;
        write_vec[write_offset++] = value/(float)ctxt_cnts_[context_it->second].second * base_prob;
      }
      if(*ngram.rbegin() == GlobalVars::curr_word) {
        // cerr << "write_vec[" << write_offset-1 << "] == " << write_vec[write_offset-1] << endl;
      }
    }
    if(j != 0) {
      this_ngram.insert(this_ngram.begin(), ngram[ngram.size()-ctxt_pos_[j-1]-1]);
      this_ctxt.insert(this_ctxt.begin(), ngram[ngram.size()-ctxt_pos_[j-1]-1]);
    }
  }
  // Convert into heuristics if necessary
  if(heuristics_) {
    if(smoothing_ == SMOOTH_MABS || smoothing_ == SMOOTH_MKN) {    
      vector<float> ctxts((ctxt_pos_.size()+1)*4);
      calc_ctxt_feats(this_ctxt, &ctxts[0]);
      float val = 0.f;
      float left = 1.0;
      for(int i = temp_vec.size() - 1; i >= 0; i--) {
        if(ctxts[i*4] == 1.0) continue;
        // Discounted divided by total == amount for this dist
        float my_prob = exp(ctxts[i*4+3]-ctxts[i*4+1]);
        if(my_prob > 1) THROW_ERROR("Discount exceeding 1: " << my_prob);
        val += left * my_prob * temp_vec[i];
        left *= (1-my_prob);
      }
      val += left*uniform_prob;
      if(val < 0 || val > 1) {
        cerr << "this_ctxt: " << this_ctxt << endl;
        cerr << "ctxts: " << ctxts << endl;
        cerr << "temp_vec: " << temp_vec << endl;
        cerr << "val=" << val << ", left=" << left << endl;
        THROW_ERROR("Illegal probability value");
      }
      trg_dense[dense_offset++] = val;
    } else {
      THROW_ERROR("Heuristics are only implemented for discounting");
    }
  }
}

// Read/write model. If dict is null, use numerical ids, otherwise strings.
#define DIST_NGRAM_VERSION "distngram_v2"
void DistNgram::write(DictPtr dict, std::ostream & out) const {
  out << DIST_NGRAM_VERSION << '\n';
  if(smoothing_ != SMOOTH_LIN) {
    out << "discounts " << discounts_.size() << '\n';  
    for(auto & discount : discounts_)
      out << discount[1] << ' ' << discount[2] << ' ' << discount[3] << '\n';
    out << '\n';
  }
  out << "mapping " << mapping_.size() << ' ' << ctxt_cnts_.size() << '\n';
  for(auto & map_elem : mapping_) {
    out << PrintWords(*dict, map_elem.first) << '\t';
    if(map_elem.first.size() == ngram_len_) {
      out << map_elem.second;
    } else {
      out << ctxt_cnts_[map_elem.second].first << ' ' << ctxt_cnts_[map_elem.second].second << ' ' << ctxt_cnts_[map_elem.second].third;
      if(disc_ctxt_cnts_.size() != 0) out << ' ' << disc_ctxt_cnts_[map_elem.second];
    }
    out << '\n';
  }
  out << '\n';
}

inline void getline_or_die(std::istream & in, std::string & line) {
  if(!getline(in, line)) THROW_ERROR("Premature end at DistNgram");
}
inline void getline_expected(std::istream & in, const std::string & expected) {
  string line;
  if(!getline(in, line)) THROW_ERROR("Premature end at DistNgram");
  if(line != expected) THROW_ERROR("Did not get expected line at DistNgram: " << line << " != " << expected);
}
void DistNgram::read(DictPtr dict, std::istream & in) {
  vector<string> strs;
  string line, strid;
  int size, size2;
  int cnt1, cnt2, cnt3;
  float cnt4;
  getline_expected(in, DIST_NGRAM_VERSION);
  // Get the discounts
  if(smoothing_ != SMOOTH_LIN) {
    getline_or_die(in, line);
    istringstream iss(line); iss >> strid >> size;
    if(strid != "discounts" || size != ngram_len_) THROW_ERROR("Bad format of discounts: " << line << endl);
    discounts_.resize(size, vector<float>(4));
    for(int i = 0; i < size; i++) {
      getline_or_die(in, line); istringstream iss2(line);
      iss2 >> discounts_[i][1] >> discounts_[i][2] >> discounts_[i][3];
    }
    getline_expected(in, "");
  }
  // Get the mapping
  {
    getline_or_die(in, line);
    istringstream iss(line); iss >> strid >> size >> size2;
    if(strid != "mapping") THROW_ERROR("Bad format of mapping: " << line << endl);
    mapping_.reserve(size*1.1);
    ctxt_cnts_.reserve(size2);
    if(smoothing_ != SMOOTH_LIN) disc_ctxt_cnts_.reserve(size2);
    for(int i = 0; i < size; i++) {
      getline_or_die(in, line);
      boost::split(strs, line, boost::is_any_of("\t"));
      if(strs.size() != 2) THROW_ERROR("Expecting two columns: " << line << endl);
      Sentence ngram = ParseWords(*dict, strs[0], false);
      for(WordId wid : ngram) if(wid == -1) THROW_ERROR("Out-of-vocabulary word found in one hot model: " << line);
      if(ngram.size() == ngram_len_) {
        mapping_[ngram] = stoi(strs[1]);
      } else {
        mapping_[ngram] = ctxt_cnts_.size();
        istringstream iss(strs[1]);
        iss >> cnt1 >> cnt2 >> cnt3; ctxt_cnts_.push_back(DistNgramCounts(cnt1,cnt2,cnt3));
        if(smoothing_ != SMOOTH_LIN) { iss >> cnt4; disc_ctxt_cnts_.push_back(cnt4); }
      }
    }
    getline_expected(in, "");
  }
}
