#pragma once


#include "dynet/dynet.h"
#include "dynet/rnn.h"
#include "lamtram/extern-calculator.h"

namespace dynet {

class Model;

  
}

using namespace dynet;

namespace lamtram {

struct RNNCONDBuilder : public RNNBuilder {
  RNNCONDBuilder() = default;

 virtual dynet::expr::Expression add_input_withContext( const Expression & x, Expression & attention_context,
        const dynet::expr::Expression & align_sum_in,
        bool train,
        dynet::ComputationGraph & cg,
        std::vector<dynet::expr::Expression> & align_out,
        dynet::expr::Expression & align_sum_out);

 protected:

  ExternCalculatorPtr att_;
  Expression const * align_sum_in_;
  bool train_;
  ComputationGraph* cg_;
  std::vector<Expression>* align_out_;
  Expression * align_sum_out_;
  Expression attention_context_;
  
};

} // namespace lamtram

