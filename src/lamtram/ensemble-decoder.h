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
    EnsembleDecoderHyp(cnn::real score, const std::vector<std::vector<cnn::expr::Expression> > & states, const Sentence & sent) :
        score_(score), states_(states), sent_(sent) { }

    cnn::real GetScore() const { return score_; }
    const std::vector<std::vector<cnn::expr::Expression> > & GetStates() const { return states_; }
    const Sentence & GetSentence() const { return sent_; }

protected:

    cnn::real score_;
    std::vector<std::vector<cnn::expr::Expression> > states_;
    Sentence sent_;

};

typedef std::shared_ptr<EnsembleDecoderHyp> EnsembleDecoderHypPtr;

class EnsembleDecoder {

public:
    EnsembleDecoder(const std::vector<EncoderDecoderPtr> & encdecs,
                    const std::vector<EncoderAttentionalPtr> & encatts,
                    const std::vector<NeuralLMPtr> & lms, int pad);
    ~EnsembleDecoder() {}

    void CalcSentLL(const Sentence & sent_src, const Sentence & sent_trg, LLStats & ll);
    Sentence Generate(const Sentence & sent_src);

    std::vector<std::vector<cnn::expr::Expression> > GetInitialStates(const Sentence & sent_src, cnn::ComputationGraph & cg);
    
    // Ensemble together probabilities or log probabilities for a single word
    cnn::expr::Expression EnsembleProbs(const std::vector<cnn::expr::Expression> & in, cnn::ComputationGraph & cg);
    cnn::expr::Expression EnsembleLogProbs(const std::vector<cnn::expr::Expression> & in, cnn::ComputationGraph & cg);
    cnn::expr::Expression EnsembleSingleProb(const std::vector<cnn::expr::Expression> & in, int word, cnn::ComputationGraph & cg);
    cnn::expr::Expression EnsembleSingleLogProb(const std::vector<cnn::expr::Expression> & in, int word, cnn::ComputationGraph & cg);

    cnn::real GetWordPen() const { return word_pen_; }
    std::string GetEnsembleOperation() const { return ensemble_operation_; }
    void SetWordPen(cnn::real word_pen) { word_pen_ = word_pen; }
    void SetEnsembleOperation(const std::string & ensemble_operation) { ensemble_operation_ = ensemble_operation; }

    int GetBeamSize() const { return beam_size_; }
    void SetBeamSize(int beam_size) { beam_size_ = beam_size; }

protected:
    std::vector<EncoderDecoderPtr> encdecs_;
    std::vector<EncoderAttentionalPtr> encatts_;
    std::vector<NeuralLMPtr> lms_;
    std::vector<ExternCalculatorPtr> externs_;
    cnn::real word_pen_;
    int pad_;
    int unk_id_;
    int size_limit_;
    int beam_size_;
    std::string ensemble_operation_;

};

}
