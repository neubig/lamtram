#pragma once

#include <lamtram/encoder-decoder.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/neural-lm.h>
#include <lamtram/extern-calculator.h>
#include <cnn/tensor.h>
#include <cnn/cnn.h>
#include <vector>

namespace lamtram {

class EnsembleDecoderHyp {
public:
    EnsembleDecoderHyp(float score, const std::vector<std::vector<cnn::expr::Expression> > & states, const std::vector<cnn::expr::Expression> & externs, const std::vector<cnn::expr::Expression> & sums, const Sentence & sent, const Sentence & align) :
        score_(score), states_(states), externs_(externs), sums_(sums), sent_(sent), align_(align) { }

    float GetScore() const { return score_; }
    const std::vector<std::vector<cnn::expr::Expression> > & GetStates() const { return states_; }
    const std::vector<cnn::expr::Expression> & GetExterns() const { return externs_; }
    const std::vector<cnn::expr::Expression> & GetSums() const { return sums_; }
    const Sentence & GetSentence() const { return sent_; }
    const Sentence & GetAlignment() const { return align_; }

protected:

    float score_;
    std::vector<std::vector<cnn::expr::Expression> > states_;
    std::vector<cnn::expr::Expression> externs_;
    std::vector<cnn::expr::Expression> sums_;
    Sentence sent_;
    Sentence align_;

};

typedef std::shared_ptr<EnsembleDecoderHyp> EnsembleDecoderHypPtr;
inline bool operator<(const EnsembleDecoderHypPtr & lhs, const EnsembleDecoderHypPtr & rhs) {
  if(lhs->GetScore() != rhs->GetScore()) return lhs->GetScore() > rhs->GetScore();
  return lhs->GetSentence() < rhs->GetSentence();
}

class EnsembleDecoder {

public:
    EnsembleDecoder(const std::vector<EncoderDecoderPtr> & encdecs,
                    const std::vector<EncoderAttentionalPtr> & encatts,
                    const std::vector<NeuralLMPtr> & lms);
    ~EnsembleDecoder() {}

    template <class OutSent, class OutLL, class OutWords>
    void CalcSentLL(const Sentence & sent_src, const OutSent & sent_trg, OutLL & ll, OutWords & words);

    EnsembleDecoderHypPtr Generate(const Sentence & sent_src);
    std::vector<EnsembleDecoderHypPtr> GenerateNbest(const Sentence & sent_src, int nbest);

    std::vector<std::vector<cnn::expr::Expression> > GetInitialStates(const Sentence & sent_src, cnn::ComputationGraph & cg);
    
    template <class Sent, class Stat, class WordLik>
    void AddLik(const Sent & sent, const cnn::expr::Expression & expr, const std::vector<cnn::expr::Expression> & exprs, Stat & ll, WordLik & wordll);

    // Ensemble together probabilities or log probabilities for a single word
    cnn::expr::Expression EnsembleProbs(const std::vector<cnn::expr::Expression> & in, cnn::ComputationGraph & cg);
    cnn::expr::Expression EnsembleLogProbs(const std::vector<cnn::expr::Expression> & in, cnn::ComputationGraph & cg);

    // Ensemble log probs for a single value
    template <class Sent>
    cnn::expr::Expression EnsembleSingleProb(const std::vector<cnn::expr::Expression> & in, const Sent & sent, int loc, cnn::ComputationGraph & cg);
    template <class Sent>
    cnn::expr::Expression EnsembleSingleLogProb(const std::vector<cnn::expr::Expression> & in, const Sent & sent, int loc, cnn::ComputationGraph & cg);

    float GetWordPen() const { return word_pen_; }
    float GetUnkPen() const { return unk_pen_; }
    std::string GetEnsembleOperation() const { return ensemble_operation_; }
    void SetWordPen(float word_pen) { word_pen_ = word_pen; }
    void SetUnkPen(float unk_pen) { unk_pen_ = unk_pen; }
    void SetEnsembleOperation(const std::string & ensemble_operation) { ensemble_operation_ = ensemble_operation; }

    int GetBeamSize() const { return beam_size_; }
    void SetBeamSize(int beam_size) { beam_size_ = beam_size; }
    int GetSizeLimit() const { return size_limit_; }
    void SetSizeLimit(int size_limit) { size_limit_ = size_limit; }

protected:
    std::vector<EncoderDecoderPtr> encdecs_;
    std::vector<EncoderAttentionalPtr> encatts_;
    std::vector<NeuralLMPtr> lms_;
    std::vector<ExternCalculatorPtr> externs_;
    float word_pen_;
    float unk_pen_, unk_log_prob_;
    int unk_id_;
    int size_limit_;
    int beam_size_;
    std::string ensemble_operation_;

};

}
