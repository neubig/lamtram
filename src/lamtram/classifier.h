#pragma once

#include <cnn/model.h>
#include <cnn/expr.h>
#include <vector>
#include <iostream>
#include <memory>

namespace lamtram {

// A class for feed-forward neural network classifiers
class Classifier {

public:

    Classifier(int input_size, int label_size, const std::string & layers, cnn::Model & mod);
    ~Classifier() { }

    // Calculate the log likelihood
    template <class OutputData>
    cnn::expr::Expression BuildGraph(const cnn::expr::Expression & input, const OutputData & label,
                                     bool train,
                                     cnn::ComputationGraph & cg) const;

    // Calculate the probabilities from the model, or predict
    template <class SoftmaxOp>
    cnn::expr::Expression Forward(const cnn::expr::Expression & input,
                                  cnn::ComputationGraph & cg) const;

    // Accessors
    int GetLabelSize() const { return label_size_; }
    int GetInputSize() const { return input_size_; }

    // Index the parameters in a computation graph
    void NewGraph(cnn::ComputationGraph & cg);
    
    static Classifier* Read(std::istream & in, cnn::Model & mod);
    void Write(std::ostream & out);

protected:

    int input_size_; 
    int label_size_;
    std::string layer_str_;

    std::vector<cnn::Parameter> p_W_; // Layer weights
    std::vector<cnn::Parameter> p_b_; // Layer bias

    std::vector<cnn::expr::Expression> i_W_; // Layer weights
    std::vector<cnn::expr::Expression> i_b_; // Layer bias

private:
    // A pointer to the current computation graph.
    // This is only used for sanity checking to make sure NewGraph
    // is called before trying to do anything that requires it.
    cnn::ComputationGraph * curr_graph_;

};

typedef std::shared_ptr<Classifier> ClassifierPtr;

}
