#pragma once

#include <cnn/expr.h>
#include <lamtram/softmax-base.h>
#include <lamtram/dist-base.h>

namespace cnn { class Parameter; }

namespace lamtram {

// An interface to a class that takes a vector as input
// (potentially batched) and calculates a probability distribution
// over words
class SoftmaxDiff : public SoftmaxBase {

public:
  SoftmaxDiff(const std::string & sig, int input_size, const DictPtr & vocab, cnn::Model & mod);
  ~SoftmaxDiff() { };

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

protected:

  cnn::expr::Expression CalcLossExpr(cnn::expr::Expression & in, const std::vector<float> & ctxt_dist, WordId wid, bool train);
  cnn::expr::Expression CalcLossExpr(cnn::expr::Expression & in, const std::vector<float> & ctxt_dist_batched, const std::vector<unsigned> & wids, bool train);

  void LoadDists(int id);

  void CalcDists(const Sentence & ngram, std::vector<float> & ctxt_dist);
  void CalcAllDists(const Sentence & ctxt_ngram, std::vector<float> & ctxt_dist);

  int vocab_size_;

  int finished_words_, drop_words_;

  cnn::Parameter p_sm_W_, p_sm_b_; // Softmax weights

  cnn::expr::Expression i_sm_W_, i_sm_b_;

  float dropout_; // How much to drop out the dense distributions (at training)
  DistPtr dist_ptr_;
  int dist_id_;

  std::vector<std::vector<float> > cache_;
  std::vector<std::string> wildcards_;
  std::vector<std::string> dist_files_;

};

}
