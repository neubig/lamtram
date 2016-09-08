#pragma once


#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "lamtram/gru-cond.h"
#include "lamtram/extern-calculator.h"

namespace cnn {

class Model;

  
}

using namespace cnn;

namespace lamtram {

struct GRUCONDBuilder : public RNNBuilder {
  GRUCONDBuilder() = default;
  explicit GRUCONDBuilder(unsigned layers,
                      unsigned input_dim,
                      unsigned input_2_dim,
                      unsigned hidden_dim,
                      cnn::Model* model,ExternCalculatorPtr & att);
  Expression back() const override { return (cur == -1? h0.back() : h[cur].back()); }
  std::vector<Expression> final_h() const override { return (h.size() == 0 ? h0 : h.back()); }
  std::vector<Expression> final_s() const override { return final_h(); }
  std::vector<Expression> get_h(RNNPointer i) const override { return (i == -1 ? h0 : h[i]); }
  std::vector<Expression> get_s(RNNPointer i) const override { return get_h(i); }
  unsigned num_h0_components() const override { return layers; }
  void copy(const RNNBuilder & params) override;

  void init_parameters(int layer,int index, const std::vector<float>& vec);
  
  void SetAttention(ExternCalculatorPtr att) {att_ = att;};

 protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
  Expression add_input_impl(int prev, const Expression& x) override;

  // first index is layer, then ...
  std::vector<std::vector<Parameter>> params;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h;

  // initial values of h at each layer
  // - default to zero matrix input
  std::vector<Expression> h0;

  unsigned hidden_dim;
  unsigned layers;
  
  ExternCalculatorPtr & att_;
};

} // namespace lamtram

