#pragma once

#include <dynet/model.h>
#include <dynet/expr.h>
#include <vector>
#include <iostream>
#include <memory>

namespace lamtram {

// A class for feed-forward neural network classifiers
class Classifier {

public:

    Classifier(int input_size, int label_size, const std::string & layers, const std::string & smsig, dynet::Model & mod);
    ~Classifier() { }

    // Calculate the log likelihood
    template <class OutputData>
    dynet::expr::Expression BuildGraph(const dynet::expr::Expression & input, const OutputData & label,
                                     bool train,
                                     dynet::ComputationGraph & cg) const;

    // Calculate the probabilities from the model, or predict
    template <class SoftmaxOp>
    dynet::expr::Expression Forward(const dynet::expr::Expression & input,
                                  dynet::ComputationGraph & cg) const;

    // Accessors
    int GetLabelSize() const { return label_size_; }
    int GetInputSize() const { return input_size_; }

    // Index the parameters in a computation graph
    void NewGraph(dynet::ComputationGraph & cg);
    
    static Classifier* Read(std::istream & in, dynet::Model & mod);
    void Write(std::ostream & out);

    // Setters
    void SetDropout(float dropout) { dropout_ = dropout; }

protected:

    int input_size_; 
    int label_size_;
    std::string layer_str_;
    std::string smsig_;

    std::vector<dynet::Parameter> p_W_; // Layer weights
    std::vector<dynet::Parameter> p_b_; // Layer bias

    std::vector<dynet::expr::Expression> i_W_; // Layer weights
    std::vector<dynet::expr::Expression> i_b_; // Layer bias

private:
    // A pointer to the current computation graph.
    // This is only used for sanity checking to make sure NewGraph
    // is called before trying to do anything that requires it.
    dynet::ComputationGraph * curr_graph_;

    // Dropout rate
    float dropout_;

};

typedef std::shared_ptr<Classifier> ClassifierPtr;

}
