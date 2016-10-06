#pragma once

#include <lamtram/sentence.h>
#include <lamtram/ll-stats.h>
#include <lamtram/linear-encoder.h>
#include <lamtram/neural-lm.h>
#include <dynet/dynet.h>
#include <vector>
#include <iostream>

namespace dynet {
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
                   dynet::Model & model);
    ~EncoderDecoder() { }

    // Build the computation graph for the sentence including loss
    dynet::expr::Expression BuildSentGraph(const Sentence & sent_src,
                                         const Sentence & sent_trg,
                                         const Sentence & cache_trg,
                                         const float * weight,
                                         float samp_percent,
                                         bool train,
                                         dynet::ComputationGraph & cg,
                                         LLStats & ll);
    dynet::expr::Expression BuildSentGraph(const std::vector<Sentence> & sent_src,
                                         const std::vector<Sentence> & sent_trg,
                                         const std::vector<Sentence> & cache_trg,
                                         const std::vector<float> * weights,
                                         float samp_percent,
                                         bool train,
                                         dynet::ComputationGraph & cg,
                                         LLStats & ll);

    // Sample sentences and return an expression of the vector of probabilities
    dynet::expr::Expression SampleTrgSentences(const Sentence & sent_src,
                                             const Sentence * sent_trg,   
                                             int num_samples,
                                             int max_len,
                                             bool train,
                                             dynet::ComputationGraph & cg,
                                             std::vector<Sentence> & samples);

    template <class SentData>
    std::vector<dynet::expr::Expression> GetEncodedState(
                                        const SentData & sent_src, bool train, dynet::ComputationGraph & cg);

    // Reading/writing functions
    static EncoderDecoder* Read(const DictPtr & vocab_src, const DictPtr & vocab_trg, std::istream & in, dynet::Model & model);
    void Write(std::ostream & out);

    // Index the parameters in a computation graph
    void NewGraph(dynet::ComputationGraph & cg);

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

    // Setters
    void SetDropout(float dropout) {
      for(auto & enc : encoders_) enc->SetDropout(dropout);
      decoder_->SetDropout(dropout);
    }

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
    dynet::Parameter p_enc2dec_W_; // Encoder to decoder weights
    dynet::Parameter p_enc2dec_b_; // Encoder to decoder bias
    
    // Indicies in the current computation graph for each parameter
    dynet::expr::Expression i_enc2dec_W_;
    dynet::expr::Expression i_enc2dec_b_;

private:
    // A pointer to the current computation graph.
    // This is only used for sanity checking to make sure NewGraph
    // is called before trying to do anything that requires it.
    dynet::ComputationGraph * curr_graph_;

};

typedef std::shared_ptr<EncoderDecoder> EncoderDecoderPtr;

}
