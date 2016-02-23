#pragma once

#include <lamtram/sentence.h>
#include <lamtram/ll-stats.h>
#include <lamtram/builder-factory.h>
#include <cnn/cnn.h>
#include <cnn/expr.h>
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

// A class for feed-forward neural network LMs
class LinearEncoder {

public:

    // Create a new LinearEncoder and add it to the existing model
    LinearEncoder(int vocab_size, int wordrep_size,
             const std::string & hidden_spec, int unk_id,
             cnn::Model & model);
    ~LinearEncoder() { }

    // Build the computation graph for the sentence including loss
    cnn::expr::Expression BuildSentGraph(const Sentence & sent, bool train, cnn::ComputationGraph & cg);

    // Reading/writing functions
    static LinearEncoder* Read(std::istream & in, cnn::Model & model);
    void Write(std::ostream & out);

    // Index the parameters in a computation graph
    void NewGraph(cnn::ComputationGraph & cg);

    // Get the last hidden layers of the encoder
    std::vector<cnn::expr::Expression> GetFinalHiddenLayers() const;

    // Clone the parameters of another linear encoder
    void CopyParameters(const LinearEncoder & enc);

    // Accessors
    int GetVocabSize() const { return vocab_size_; }
    int GetWordrepSize() const { return wordrep_size_; }
    int GetUnkId() const { return unk_id_; }
    int GetNumLayers() const { return hidden_spec_.layers; }
    int GetNumNodes() const { return hidden_spec_.nodes; }
    const std::vector<cnn::expr::Expression> & GetWordStates() const { return word_states_; }

    void SetReverse(bool reverse) { reverse_ = reverse; }

protected:

    // Variables
    int vocab_size_, wordrep_size_, unk_id_;
    BuilderSpec hidden_spec_;

    // Whether to reverse or not
    bool reverse_;

    // Pointers to the parameters
    cnn::LookupParameters* p_wr_W_; // Wordrep weights

    // The RNN builder
    BuilderPtr builder_;

    // This records the last set of word states acquired during BuildSentGraph
    std::vector<cnn::expr::Expression> word_states_;

private:
    // A pointer to the current computation graph.
    // This is only used for sanity checking to make sure NewGraph
    // is called before trying to do anything that requires it.
    cnn::ComputationGraph * curr_graph_;

};

typedef std::shared_ptr<LinearEncoder> LinearEncoderPtr;

}
