#pragma once

#include <lamtram/sentence.h>
#include <lamtram/linear-encoder.h>
#include <lamtram/classifier.h>
#include <lamtram/ll-stats.h>
#include <lamtram/dict-utils.h>
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

// A class for feed-forward neural network LMs
class EncoderClassifier {

public:

    // Create a new EncoderClassifier and add it to the existing model
    EncoderClassifier(const std::vector<LinearEncoderPtr> & encoders,
                   const ClassifierPtr & classifier,
                   cnn::Model & model);
    ~EncoderClassifier() { }

    // Encode the input sentence as a vector to be input to the classifier
    cnn::expr::Expression GetEncodedState(
                      const Sentence & sent_src, bool train, cnn::ComputationGraph & cg) const;

    // Build the computation graph for the sentence including loss
    cnn::expr::Expression BuildSentGraph(const Sentence & sent_src, int trg,
                                         bool train,
                                         cnn::ComputationGraph & cg, LLStats & ll) const;

    // Calculate the probabilities from the model, or predict
    template <class SoftmaxOp>
    cnn::expr::Expression Forward(const Sentence & sent_src, 
                                  bool train,
                                  cnn::ComputationGraph & cg) const;

    // Reading/writing functions
    static EncoderClassifier* Read(const DictPtr & vocab_src, const DictPtr & vocab_trg, std::istream & in, cnn::Model & model);
    void Write(std::ostream & out);

    // Index the parameters in a computation graph
    void NewGraph(cnn::ComputationGraph & cg);

    // Information functions
    static bool HasSrcVocab() { return true; }
    static std::string ModelID() { return "enccls"; }

    // Accessors
    const Classifier & GetClassifier() const { return *classifier_; }
    const ClassifierPtr & GetClassifierPtr() const { return classifier_; }

protected:

    // Vectors
    std::vector<LinearEncoderPtr> encoders_;
    ClassifierPtr classifier_;

    // Parameters
    cnn::Parameters* p_enc2cls_W_; // Encoder to classifier weights
    cnn::Parameters* p_enc2cls_b_; // Encoder to classifier bias
    
    // Indicies in the current computation graph for each parameter
    cnn::expr::Expression i_enc2cls_W_;
    cnn::expr::Expression i_enc2cls_b_;

private:
    // A pointer to the current computation graph.
    // This is only used for sanity checking to make sure NewGraph
    // is called before trying to do anything that requires it.
    cnn::ComputationGraph * curr_graph_;

};

typedef std::shared_ptr<EncoderClassifier> EncoderClassifierPtr;

}
