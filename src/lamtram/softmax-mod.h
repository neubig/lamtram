#pragma once

#include <dynet/expr.h>
#include <lamtram/softmax-base.h>
#include <lamtram/dist-base.h>

namespace dynet { struct Parameter; }

namespace lamtram {

// An interface to a class that takes a vector as input
// (potentially batched) and calculates a probability distribution
// over words
class SoftmaxMod : public SoftmaxBase {

public:
  SoftmaxMod(const std::string & sig, int input_size, const DictPtr & vocab, dynet::Model & mod);
  ~SoftmaxMod() { };

  // A pair of a context and distribution values
  typedef std::pair<std::vector<float>, std::vector<float> > CtxtDist;

  // Create a new graph
  virtual void NewGraph(dynet::ComputationGraph & cg) override;

  // Calculate training loss for one word
  virtual dynet::expr::Expression CalcLoss(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ngram, bool train) override;
  // Calculate training loss for multiple words
  virtual dynet::expr::Expression CalcLoss(dynet::expr::Expression & in, dynet::expr::Expression & prior, const std::vector<Sentence> & ngrams, bool train) override;

  // Calculate loss using cached info
  virtual dynet::expr::Expression CalcLossCache(dynet::expr::Expression & in, dynet::expr::Expression & prior, int cache_id, const Sentence & ngram, bool train) override;
  virtual dynet::expr::Expression CalcLossCache(dynet::expr::Expression & in, dynet::expr::Expression & prior, const std::vector<int> & cache_ids, const std::vector<Sentence> & ngrams, bool train) override;
  
  // Calculate the full probability distribution
  virtual dynet::expr::Expression CalcProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ctxt, bool train) override;
  virtual dynet::expr::Expression CalcProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const std::vector<Sentence> & ctxt, bool train) override;
  virtual dynet::expr::Expression CalcLogProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ctxt, bool train) override;
  virtual dynet::expr::Expression CalcLogProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const std::vector<Sentence> & ctxt, bool train) override;
  virtual dynet::expr::Expression CalcProbCache(dynet::expr::Expression & in, dynet::expr::Expression & prior, int cache_id,               const Sentence & ctxt, bool train) override;
  virtual dynet::expr::Expression CalcProbCache(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & cache_ids, const std::vector<Sentence> & ctxt, bool train) override;
  virtual dynet::expr::Expression CalcLogProbCache(dynet::expr::Expression & in, dynet::expr::Expression & prior, int cache_id,               const Sentence & ctxt, bool train) override;
  virtual dynet::expr::Expression CalcLogProbCache(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & cache_ids, const std::vector<Sentence> & ctxt, bool train) override;

  virtual void Cache(const std::vector<Sentence> & sents, const std::vector<int> & set_ids, std::vector<Sentence> & cache_ids) override;

  virtual void UpdateFold(int fold_id) override { LoadDists(fold_id); }  

protected:

  dynet::expr::Expression CalcLossExpr(dynet::expr::Expression & in, dynet::expr::Expression & prior, const CtxtDist & ctxt_dist, WordId wid, bool train);
  dynet::expr::Expression CalcLossExpr(dynet::expr::Expression & in, dynet::expr::Expression & prior, const CtxtDist & ctxt_dist_batched, const std::vector<unsigned> & wids, bool train);

  void LoadDists(int id);

  void CalcDists(const Sentence & ngram, CtxtDist & ctxt_dist);
  void CalcAllDists(const Sentence & ctxt_ngram, CtxtDist & ctxt_dist);
  void CalcAllDists(const std::vector<Sentence> & ctxt_ngram, CtxtDist & ctxt_dist);

  int num_dist_, num_ctxt_;

  int finished_words_, drop_words_;

  dynet::Parameter p_sms_W_, p_smd_W_; // Softmax weights
  dynet::Parameter p_sms_b_, p_smd_b_; // Softmax bias

  dynet::expr::Expression i_sms_W_, i_smd_W_;
  dynet::expr::Expression i_sms_b_, i_smd_b_;

  float dropout_; // How much to drop out the dense distributions (at training)
  std::vector<DistPtr> dist_ptrs_;
  int dist_id_;

  

  std::vector<CtxtDist> cache_;
  std::vector<std::string> wildcards_;
  std::vector<std::vector<std::string> > dist_files_;

};

}
