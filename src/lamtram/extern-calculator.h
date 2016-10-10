#pragma once

#include <lamtram/sentence.h>
#include <dynet/tensor.h>
#include <vector>
#include <memory>

namespace dynet {
struct ComputationGraph;
}

namespace lamtram {

class ExternCalculator {
public:

    ExternCalculator(int context_size) : context_size_(context_size) { }
    virtual ~ExternCalculator() { }

    virtual void InitializeSentence(const Sentence & sent, bool train, dynet::ComputationGraph & cg) { }
    virtual void InitializeSentence(const std::vector<Sentence> & sent, bool train, dynet::ComputationGraph & cg) { }

    // Create a variable encoding the context
    virtual dynet::expr::Expression CreateContext(
        // const Sentence & sent, int loc,
        const std::vector<dynet::expr::Expression> & state_in,
        const dynet::expr::Expression & align_sum_in,
        bool train,
        dynet::ComputationGraph & cg,
        std::vector<dynet::expr::Expression> & align_out,
        dynet::expr::Expression & align_sum_out) const = 0;

    // Calculate the prior over the inputs
    virtual dynet::expr::Expression CalcPrior(
        const dynet::expr::Expression & align_vec) const { return dynet::expr::Expression(); };

    virtual dynet::expr::Expression GetEmptyContext(dynet::ComputationGraph & cg) const = 0;

    int GetSize() const { return context_size_; }

protected:
    int context_size_;

};

typedef std::shared_ptr<ExternCalculator> ExternCalculatorPtr;

}
