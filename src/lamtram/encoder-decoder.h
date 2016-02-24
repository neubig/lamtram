#pragma once

#include <lamtram/sentence.h>
#include <lamtram/ll-stats.h>
#include <lamtram/linear-encoder.h>
#include <lamtram/neural-lm.h>
#include <cnn/cnn.h>
#include <vector>
#include <iostream>

namespace cnn {
class Model;
struct ComputationGraph;
struct Parameter;
struct RNNBuilder;
}

namespace lamtram {

// A class for feed-forward neural network LMs
class EncoderDecoder {

public:

    // Create a new EncoderDecoder and add it to the existing model
    EncoderDecoder(const std::vector<LinearEncoderPtr> & encoders,
                   const NeuralLMPtr & decoder,
                   cnn::Model & model);
    ~EncoderDecoder() { }

    // Build the computation graph for the sentence including loss
    cnn::expr::Expression BuildSentGraph(int sent_id,
                                         const Sentence & sent_src, const Sentence & sent_trg,
                                         bool train,
                                         cnn::ComputationGraph & cg, LLStats & ll);
    std::vector<cnn::expr::Expression> GetEncodedState(
                                        const Sentence & sent_src, bool train, cnn::ComputationGraph & cg);

    // Reading/writing functions
    static EncoderDecoder* Read(const DictPtr & vocab_src, const DictPtr & vocab_trg, std::istream & in, cnn::Model & model);
    void Write(std::ostream & out);

    // Index the parameters in a computation graph
    void NewGraph(cnn::ComputationGraph & cg);

    // Information functions
    static bool HasSrcVocab() { return true; }
    static std::string ModelID() { return "encdec"; }

    // Accessors
    const NeuralLM & GetDecoder() const { return *decoder_; }
    const NeuralLMPtr & GetDecoderPtr() const { return decoder_; }
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
    std::vector<LinearEncoderPtr> encoders_;
    NeuralLMPtr decoder_;

    // Parameters
    cnn::Parameter p_enc2dec_W_; // Encoder to decoder weights
    cnn::Parameter p_enc2dec_b_; // Encoder to decoder bias
    
    // Indicies in the current computation graph for each parameter
    cnn::expr::Expression i_enc2dec_W_;
    cnn::expr::Expression i_enc2dec_b_;

private:
    // A pointer to the current computation graph.
    // This is only used for sanity checking to make sure NewGraph
    // is called before trying to do anything that requires it.
    cnn::ComputationGraph * curr_graph_;

};

typedef std::shared_ptr<EncoderDecoder> EncoderDecoderPtr;

}
