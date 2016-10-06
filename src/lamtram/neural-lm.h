#pragma once

#include <lamtram/sentence.h>
#include <lamtram/ll-stats.h>
#include <lamtram/builder-factory.h>
#include <lamtram/softmax-base.h>
#include <lamtram/dict-utils.h>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <vector>
#include <iostream>

namespace dynet {
class Model;
struct ComputationGraph;
struct LookupParameter;
struct RNNBuilder;
}

namespace lamtram {

class ExternCalculator;

// A class for feed-forward neural network LMs
class NeuralLM {

public:

    // Create a new NeuralLM and add it to the existing model
    //   vocab: the vocabulary
    //   ngram_context: number of previous words to use (should be at least one)
    //   extern_context: The size in nodes of vector containing extern context
    //     that is calculated from something other than the previous words.
    //     Can be set to zero if this doesn't apply.
    //   extern_feed: Whether to feed the previous external context back in
    //   wordrep_size: The size of the word representations.
    //   unk_id: The ID of unknown words.
    //   softmax_sig: A signature indicating the type of softmax to use
    //   model: The model in which to store the parameters.
    NeuralLM(const DictPtr & vocab, int ngram_context, int extern_context,
             bool extern_feed,
             int wordrep_size, const BuilderSpec & hidden_spec, int unk_id,
             const std::string & softmax_sig,
             dynet::Model & model);
    ~NeuralLM() { }

    // Build the computation graph for the sentence including loss
    //  REQUIRES NewGraph to be called before usage
    //   sent_id: Sentence id if caching is used, can be set to -1
    //   sent: The sentence to be used.
    //   extern_in: The id of the extern context. Ignored if extern_context=0.
    //   layer_in: The input of the hidden layer.
    //   train: Whether we're training or not (so we know whether to use dropout, etc.)
    //   cg: The computation graph.
    //   ll: The log likelihood statistics will be used here.
    dynet::expr::Expression BuildSentGraph(
                                   const Sentence & sent,
                                   const Sentence & cache_ids,
                                   const float * weight,
                                   const ExternCalculator * extern_calc,
                                   const std::vector<dynet::expr::Expression> & layer_in,
                                   float samp_percent,
                                   bool train,
                                   dynet::ComputationGraph & cg,
                                   LLStats & ll);
    dynet::expr::Expression BuildSentGraph(
                                   const std::vector<Sentence> & sent,
                                   const std::vector<Sentence> & cache_ids,
                                   const std::vector<float> * weights,
                                   const ExternCalculator * extern_calc,
                                   const std::vector<dynet::expr::Expression> & layer_in,
                                   float samp_percent,
                                   bool train,
                                   dynet::ComputationGraph & cg,
                                   LLStats & ll);

    // Acquire samples from this sentence and return their log probabilities as a vector
    dynet::expr::Expression SampleTrgSentences(
                                   const ExternCalculator * extern_calc,
                                   const std::vector<dynet::expr::Expression> & layer_in,
                                   const Sentence * answer,                                   
                                   int num_samples,
                                   int max_len,
                                   bool train,
                                   dynet::ComputationGraph & cg,
                                   std::vector<Sentence> & samples);

    // Move forward one step using the language model and return the probabilities.
    //  REQUIRES NewGraph to be called before usage
    //   sent: The sentence to be used.
    //   id: The id of the word in the sentence to be used.
    //   extern_calc: The extern calculator.
    //   log_prob: Calculate the log probability
    //   layer_in: The input of the hidden layer.
    //   extern_in: The previous extern, if necessary
    //   layer_out: The output of the hidden layer.
    //   extern_out: The next extern, if necessary
    //   cg: The computation graph.
    //   align_out: The alignments.
    template <class Sent>
    dynet::expr::Expression Forward(const Sent & sent, int id, 
                               const ExternCalculator * extern_calc,
                               bool log_prob,
                               const std::vector<dynet::expr::Expression> & layer_in,
                               const dynet::expr::Expression & extern_in,
                               const dynet::expr::Expression & extern_sum_in,
                               std::vector<dynet::expr::Expression> & layer_out,
                               dynet::expr::Expression & extern_out,
                               dynet::expr::Expression & extern_sum_out,
                               dynet::ComputationGraph & cg,
                               std::vector<dynet::expr::Expression> & align_out);

    template <class Sent>
    Sent CreateContext(const Sent & sent, int t);
    
    // Index the parameters in a computation graph
    void NewGraph(dynet::ComputationGraph & cg);

    // Reading/writing functions
    static NeuralLM* Read(const DictPtr & vocab, std::istream & in, dynet::Model & model);
    void Write(std::ostream & out);

    // Information functions
    static bool HasSrcVocab() { return false; }
    static std::string ModelID() { return "nlm"; }

    // Accessors
    int GetVocabSize() const;
    int GetNgramContext() const { return ngram_context_; }
    int GetExternalContext() const { return extern_context_; }
    int GetWordrepSize() const { return wordrep_size_; }
    int GetUnkId() const { return unk_id_; }
    int GetNumLayers() const { return hidden_spec_.layers; }
    int GetNumNodes() const { return hidden_spec_.nodes; }
    int GetLayerMultiplier() const { return hidden_spec_.multiplier; }
    SoftmaxBase & GetSoftmax() { return *softmax_; }

    // Setters
    void SetDropout(float dropout);

protected:

    // The vocabulary
    DictPtr vocab_;

    // Variables
    int ngram_context_, extern_context_;
    bool extern_feed_;
    int wordrep_size_, unk_id_;
    BuilderSpec hidden_spec_;

    // Pointers to the parameters
    dynet::LookupParameter p_wr_W_; // Wordrep weights

    // Pointer to the softmax
    SoftmaxPtr softmax_;

    // The RNN builder
    BuilderPtr builder_;

private:
    // A pointer to the current computation graph.
    // This is only used for sanity checking to make sure NewGraph
    // is called before trying to do anything that requires it.
    dynet::ComputationGraph * curr_graph_;

};

typedef std::shared_ptr<NeuralLM> NeuralLMPtr;

}
