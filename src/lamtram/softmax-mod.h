#pragma once

#include <cnn/expr.h>
#include <lamtram/softmax-base.h>
#include <lamtram/dist-base.h>

namespace cnn { class Parameter; }

namespace lamtram {

// An interface to a class that takes a vector as input
// (potentially batched) and calculates a probability distribution
// over words
class SoftmaxMod : public SoftmaxBase {

public:
  SoftmaxMod(const std::string & sig, int input_size, const DictPtr & vocab, cnn::Model & mod);
  ~SoftmaxMod() { };

  // A pair of a context and distribution values
  typedef std::pair<std::vector<float>, std::vector<float> > CtxtDist;

  // Create a new graph
  virtual void NewGraph(cnn::ComputationGraph & cg) override;

  // Calculate training loss for one word
  virtual cnn::expr::Expression CalcLoss(cnn::expr::Expression & in, const Sentence & ngram, bool train) override;
  // Calculate training loss for multiple words
  virtual cnn::expr::Expression CalcLoss(cnn::expr::Expression & in, const std::vector<Sentence> & ngrams, bool train) override;

  // Calculate loss using cached info
  virtual cnn::expr::Expression CalcLossCache(cnn::expr::Expression & in, int cache_id, const Sentence & ngram, bool train) override;
  virtual cnn::expr::Expression CalcLossCache(cnn::expr::Expression & in, const std::vector<int> & cache_ids, const std::vector<Sentence> & ngrams, bool train) override;
  
  // Calculate the full probability distribution
  virtual cnn::expr::Expression CalcProb(cnn::expr::Expression & in, const Sentence & ctxt, bool train) override;
  virtual cnn::expr::Expression CalcProb(cnn::expr::Expression & in, const std::vector<Sentence> & ctxt, bool train) override;
  virtual cnn::expr::Expression CalcLogProb(cnn::expr::Expression & in, const Sentence & ctxt, bool train) override;
  virtual cnn::expr::Expression CalcLogProb(cnn::expr::Expression & in, const std::vector<Sentence> & ctxt, bool train) override;
  virtual cnn::expr::Expression CalcProbCache(cnn::expr::Expression & in, int cache_id,               const Sentence & ctxt, bool train) override;
  virtual cnn::expr::Expression CalcProbCache(cnn::expr::Expression & in, const Sentence & cache_ids, const std::vector<Sentence> & ctxt, bool train) override;
  virtual cnn::expr::Expression CalcLogProbCache(cnn::expr::Expression & in, int cache_id,               const Sentence & ctxt, bool train) override;
  virtual cnn::expr::Expression CalcLogProbCache(cnn::expr::Expression & in, const Sentence & cache_ids, const std::vector<Sentence> & ctxt, bool train) override;

  virtual void Cache(const std::vector<Sentence> & sents, const std::vector<int> & set_ids, std::vector<Sentence> & cache_ids) override;

  virtual void UpdateFold(int fold_id) override { LoadDists(fold_id); }  

protected:

  cnn::expr::Expression CalcLossExpr(cnn::expr::Expression & in, const CtxtDist & ctxt_dist, WordId wid, bool train);
  cnn::expr::Expression CalcLossExpr(cnn::expr::Expression & in, const CtxtDist & ctxt_dist_batched, const std::vector<unsigned> & wids, bool train);

  void LoadDists(int id);

  void CalcDists(const Sentence & ngram, CtxtDist & ctxt_dist);
  void CalcAllDists(const Sentence & ctxt_ngram, CtxtDist & ctxt_dist);
  void CalcAllDists(const std::vector<Sentence> & ctxt_ngram, CtxtDist & ctxt_dist);

  int num_dist_, num_ctxt_;

  int finished_words_, drop_words_;

  cnn::Parameter p_sms_W_, p_smd_W_; // Softmax weights
  cnn::Parameter p_sms_b_, p_smd_b_; // Softmax bias

  cnn::expr::Expression i_sms_W_, i_smd_W_;
  cnn::expr::Expression i_sms_b_, i_smd_b_;

  float dropout_; // How much to drop out the dense distributions (at training)
  std::vector<DistPtr> dist_ptrs_;
  int dist_id_;

  

  std::vector<CtxtDist> cache_;
  std::vector<std::string> wildcards_;
  std::vector<std::vector<std::string> > dist_files_;

};

}
