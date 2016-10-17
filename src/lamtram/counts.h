#pragma once

#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <lamtram/dict-utils.h>
#include <lamtram/hashes.h>
#include <lamtram/training-data.h>

namespace dynet { class Dict; }

namespace lamtram {

class ContextCounts {
public:
  ContextCounts() : full_sum(0) { }
  float full_sum;
  std::unordered_map<WordId, int> cnts;

  virtual float get_denominator() const { return full_sum; }
  virtual void add_word(WordId wid, float cnt, float mod_cnt) {
    full_sum += cnt;
    cnts[wid] = cnt;
  }
};

class Counts {

protected:
  typedef std::shared_ptr<ContextCounts> ContextCountsPtr;
  std::unordered_map<Sentence, ContextCountsPtr> cnts_;

public:
  Counts() { }
  virtual ~Counts() { }

  virtual void add_count(const Sentence & idx, WordId wid, WordId last_fallback);

  virtual void finalize_count() { }
  virtual float mod_cnt(int cnt) const { return cnt; }
  virtual ContextCounts* new_counts_ptr() const { return new ContextCounts; }

  // Calculate the ctxtual features 
  virtual void calc_ctxt_feats(const Sentence & ctxt, float * fl);
  virtual size_t get_ctxt_size() { return 3; }

  // Calculate the ctxtual features 
  virtual void calc_word_dists(const Sentence & ngram,
                               float uniform_prob,
                               float unk_prob,
                               DistTarget & trg,
                               int & dense_offset) const;

  virtual void write(DictPtr dict, std::ostream & out) const;
  virtual void read(DictPtr dict, std::istream & in);
  
  const std::unordered_map<Sentence, ContextCountsPtr> & get_cnts() const { return cnts_; }

};

class ContextCountsDisc : public ContextCounts {
public:
  ContextCountsDisc() : ContextCounts(), disc_sum(0) { }
  float disc_sum;
  virtual float get_denominator() const override { return disc_sum; }
  virtual void add_word(WordId wid, float cnt, float mod_cnt) override {
    ContextCounts::add_word(wid,cnt,mod_cnt);
    disc_sum += mod_cnt;
  }
};

class CountsMabs : public Counts {

public:
  CountsMabs() { }
  virtual ~CountsMabs() { }

  virtual void finalize_count() override;

  virtual float mod_cnt(int cnt) const override;
  
  virtual ContextCounts* new_counts_ptr() const override { return new ContextCountsDisc; }
  
  virtual void calc_ctxt_feats(const Sentence & ctxt, float * fl) override;
  virtual size_t get_ctxt_size() override { return 4; }
  
  virtual void write(DictPtr dict, std::ostream & out) const override;

  virtual void read(DictPtr dict, std::istream & in) override;

protected:
  std::vector<float> discounts_;

};

class CountsMkn : public CountsMabs {

public:
  CountsMkn() { }
  virtual ~CountsMkn() { }

  virtual void add_count(const Sentence & idx, WordId wid, WordId last_fallback) override;

  virtual void finalize_count() override;

protected:
  typedef std::pair<int, std::unordered_map<WordId, std::unordered_set<int> > > ContextCountsUniq;
  typedef std::shared_ptr<ContextCountsUniq> ContextCountsUniqPtr;
  std::unordered_map<Sentence, ContextCountsUniqPtr> cnts_uniq_;

};

typedef std::shared_ptr<Counts> CountsPtr;

}
