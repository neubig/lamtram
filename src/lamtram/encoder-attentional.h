#pragma once

#include <lamtram/sentence.h>
#include <lamtram/ll-stats.h>
#include <lamtram/linear-encoder.h>
#include <lamtram/neural-lm.h>
#include <lamtram/extern-calculator.h>
#include <cnn/cnn.h>
#include <vector>
#include <iostream>

namespace cnn {
class Model;
struct ComputationGraph;
struct LookupParameters;
struct Parameters;
struct RNNBuilder;
}

namespace lamtram {

// A class to calculate extern_calcal context
class ExternAttentional : public ExternCalculator {
public:

    ExternAttentional(const std::vector<LinearEncoderPtr> & encoders,
                      int hidden_size, int state_size,
                      cnn::Model & mod);
    virtual ~ExternAttentional() { }

    // Index the parameters in a computation graph
    void NewGraph(cnn::ComputationGraph & cg);

    // Initialize the sentence with one or more sets of encoded input
    void InitializeSentence(const Sentence & sent, cnn::ComputationGraph & cg) override;

    // Create a variable encoding the context
    virtual cnn::expr::Expression CreateContext(
        const Sentence & sent, int loc,
        const std::vector<cnn::expr::Expression> & state_in,
        cnn::ComputationGraph & cg) const override;

    int GetHiddenSize() const { return hidden_size_; }
    int GetStateSize() const { return state_size_; }

    // Reading/writing functions
    static ExternAttentional* Read(std::istream & in, cnn::Model & model);
    void Write(std::ostream & out);

protected:
    std::vector<LinearEncoderPtr> encoders_;
    int hidden_size_, state_size_;

    // Parameters
    cnn::Parameters* p_ehid_h_W_;
    cnn::Parameters* p_ehid_state_W_;
    cnn::Parameters* p_e_ehid_W_;

    // Interned parameters
    cnn::expr::Expression i_ehid_h_W_;
    cnn::expr::Expression i_ehid_state_W_;
    cnn::expr::Expression i_e_ehid_W_;

    // Temporary variables
    cnn::expr::Expression i_h_;
    cnn::expr::Expression i_ehid_hpart_;
    cnn::expr::Expression i_sent_len_;

private:
    // A pointer to the current computation graph.
    // This is only used for sanity checking to make sure NewGraph
    // is called before trying to do anything that requires it.
    cnn::ComputationGraph * curr_graph_;

    // Internal storage of a vector full of ones
    std::vector<cnn::real> sent_values_;

    int sent_len_;

};

typedef std::shared_ptr<ExternAttentional> ExternAttentionalPtr;

// A class for feed-forward neural network LMs
class EncoderAttentional {

public:

    // Create a new EncoderAttentional and add it to the existing model
    EncoderAttentional(const ExternAttentionalPtr & extern_calc,
                       const NeuralLMPtr & decoder,
                       cnn::Model & model);
    ~EncoderAttentional() { }

    // Build the computation graph for the sentence including loss
    cnn::expr::Expression BuildSentGraph(const Sentence & sent_src, const Sentence & sent_trg,
                                      cnn::ComputationGraph & cg, LLStats & ll);
    std::vector<cnn::expr::Expression> GetEncodedState(
                                    const Sentence & sent_src, cnn::ComputationGraph & cg);

    // Reading/writing functions
    static EncoderAttentional* Read(std::istream & in, cnn::Model & model);
    void Write(std::ostream & out);

    // Index the parameters in a computation graph
    void NewGraph(cnn::ComputationGraph & cg);

    // Information functions
    static bool HasSrcVocab() { return true; }
    static std::string ModelID() { return "encatt"; }

    // Accessors
    const NeuralLM & GetDecoder() const { return *decoder_; }
    const NeuralLMPtr & GetDecoderPtr() const { return decoder_; }
    const ExternAttentional & GetExternAttentional() const { return *extern_calc_; }
    const ExternAttentionalPtr & GetExternAttentionalPtr() const { return extern_calc_; }
    const ExternCalculator & GetExternCalc() const { return (ExternCalculator&)*extern_calc_; }
    const ExternCalculatorPtr GetExternCalcPtr() const { return std::dynamic_pointer_cast<ExternCalculator>(extern_calc_); }
    ExternCalculatorPtr GetExternCalcPtr() { return std::dynamic_pointer_cast<ExternCalculator>(extern_calc_); }
    int GetVocabSrc() const { return vocab_src_; }
    int GetVocabTrg() const { return vocab_trg_; }
    int GetNgramContext() const { return ngram_context_; }
    int GetWordrepSize() const { return wordrep_size_; }
    int GetUnkSrc() const { return unk_src_; }
    int GetUnkTrg() const { return unk_trg_; }

protected:

    // Variables
    int vocab_src_, vocab_trg_;
    int ngram_context_, wordrep_size_;
    int unk_src_, unk_trg_;
    std::string hidden_spec_;

    // Vectors
    ExternAttentionalPtr extern_calc_;
    NeuralLMPtr decoder_;

private:
    // A pointer to the current computation graph.
    // This is only used for sanity checking to make sure NewGraph
    // is called before trying to do anything that requires it.
    cnn::ComputationGraph * curr_graph_;

};

typedef std::shared_ptr<EncoderAttentional> EncoderAttentionalPtr;

}
