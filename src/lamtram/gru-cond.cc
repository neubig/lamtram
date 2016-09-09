#include "lamtram/gru-cond.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>
#include <lamtram/macros.h>

#include "cnn/nodes.h"
#include "cnn/training.h"

using namespace std;
using namespace cnn;

namespace lamtram {

enum { X2Z, H2Z, BZ, X2R, H2R, BR, X2H, H2H, BH, X2Z2, H2Z2, BZ2, X2R2, H2R2, BR2, X2H2, H2H2, BH2};

GRUCONDBuilder::GRUCONDBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned input_2_dim,
                       unsigned hidden_dim,
                       Model* model,ExternCalculatorPtr & att) : hidden_dim(hidden_dim), layers(layers), att_(att) {
  unsigned layer_input_dim = input_dim;
  unsigned layer_input_2_dim = input_2_dim;
  
  assert(layers == 1);
  for (unsigned i = 0; i < layers; ++i) {
    // z
    Parameter p_x2z = model->add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2z = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bz = model->add_parameters({hidden_dim});

    // r
    Parameter p_x2r = model->add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2r = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_br = model->add_parameters({hidden_dim});

    // h
    Parameter p_x2h = model->add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2h = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bh = model->add_parameters({hidden_dim});

    //and the second step after attention
    // z
    Parameter p_x2z_2 = model->add_parameters({hidden_dim, layer_input_2_dim});
    Parameter p_h2z_2 = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bz_2 = model->add_parameters({hidden_dim});

    // r
    Parameter p_x2r_2 = model->add_parameters({hidden_dim, layer_input_2_dim});
    Parameter p_h2r_2 = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_br_2 = model->add_parameters({hidden_dim});

    // h
    Parameter p_x2h_2 = model->add_parameters({hidden_dim, layer_input_2_dim});
    Parameter p_h2h_2 = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bh_2 = model->add_parameters({hidden_dim});


    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next
    

    vector<Parameter> ps = {p_x2z, p_h2z, p_bz, p_x2r, p_h2r, p_br, p_x2h, p_h2h, p_bh,p_x2z_2, p_h2z_2, p_bz_2, p_x2r_2, p_h2r_2, p_br_2, p_x2h_2, p_h2h_2, p_bh_2};
    params.push_back(ps);
  }  // layers
  dropout_rate = 0.f;
}

void GRUCONDBuilder::new_graph_impl(ComputationGraph& cg) {
  param_vars.clear();
  for (unsigned i = 0; i < layers; ++i) {
    auto& p = params[i];

    // z
    Expression x2z = parameter(cg,p[X2Z]);
    Expression h2z = parameter(cg,p[H2Z]);
    Expression bz = parameter(cg,p[BZ]);

    // r
    Expression x2r = parameter(cg,p[X2R]);
    Expression h2r = parameter(cg,p[H2R]);
    Expression br = parameter(cg,p[BR]);

    // h
    Expression x2h = parameter(cg,p[X2H]);
    Expression h2h = parameter(cg,p[H2H]);
    Expression bh = parameter(cg,p[BH]);

    //and the second step after attention
    // z_2
    Expression x2z_2 = parameter(cg,p[X2Z2]);
    Expression h2z_2 = parameter(cg,p[H2Z2]);
    Expression bz_2 = parameter(cg,p[BZ2]);

    // r
    Expression x2r_2 = parameter(cg,p[X2R2]);
    Expression h2r_2 = parameter(cg,p[H2R2]);
    Expression br_2 = parameter(cg,p[BR2]);

    // h
    Expression x2h_2 = parameter(cg,p[X2H2]);
    Expression h2h_2 = parameter(cg,p[H2H2]);
    Expression bh_2 = parameter(cg,p[BH2]);


    vector<Expression> vars = {x2z, h2z, bz, x2r, h2r, br, x2h, h2h, bh,x2z_2, h2z_2, bz_2, x2r_2, h2r_2, br_2, x2h_2, h2h_2, bh_2};
    param_vars.push_back(vars);
  }
}

void GRUCONDBuilder::start_new_sequence_impl(const std::vector<Expression>& h_0) {
  h.clear();
  h0 = h_0;
  if (!h0.empty()) {
    assert (h0.size() == layers);
  }
}

Expression GRUCONDBuilder::add_input_impl(int prev, const Expression& x) {
  if(dropout_rate != 0.f)
    throw std::runtime_error("GRUCONDBuilder doesn't support dropout yet");
  const bool has_initial_state = (h0.size() > 0);
  h.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  Expression in = x;
  if(layers > 1)
    THROW_ERROR("Cond GRU with several layers not yet implemented!");
  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    Expression h_tprev;
    // prev_zero means that h_tprev should be treated as 0
    bool prev_zero = false;
    if (prev >= 0 || has_initial_state) {
      h_tprev = (prev < 0) ? h0[i] : h[prev][i];
    } else { prev_zero = true; }
    // update gate
    Expression zt;
    if (prev_zero)
      zt = affine_transform({vars[BZ], vars[X2Z], in});
    else
      zt = affine_transform({vars[BZ], vars[X2Z], in, vars[H2Z], h_tprev});
    zt = logistic(zt);
    // forget
    Expression ft = 1.f - zt;
    // reset gate
    Expression rt;
    if (prev_zero)
      rt = affine_transform({vars[BR], vars[X2R], in});
    else
      rt = affine_transform({vars[BR], vars[X2R], in, vars[H2R], h_tprev});
    rt = logistic(rt);

    // candidate activation
    Expression ct;
    Expression hs;
    if (prev_zero) {
      ct = affine_transform({vars[BH], vars[X2H], in});
      ct = tanh(ct);
      //Expression nwt = cwise_multiply(zt, ct);
      //in = ht[i] = nwt;
      hs = cwise_multiply(zt, ct);
    } else {
      Expression ght = cwise_multiply(rt, h_tprev);
      ct = affine_transform({vars[BH], vars[X2H], in, vars[H2H], ght});
      ct = tanh(ct);
      Expression nwt = cwise_multiply(zt, ct);
      Expression crt = cwise_multiply(ft, h_tprev);
      //in = ht[i] = crt + nwt;
      hs = crt + nwt;
    }
    

    //get context;
    Expression c = att_->CalcContext(hs);
    
    // update gate
    Expression zt2 = affine_transform({vars[BZ2], vars[X2Z2], c, vars[H2Z2], hs});
    zt2 = logistic(zt2);
    // forget
    Expression ft2 = 1.f - zt2;
    // reset gate
    Expression rt2 = affine_transform({vars[BR2], vars[X2R2], c, vars[H2R2], hs});
    rt2 = logistic(rt2);

    // candidate activation
    Expression ght = cwise_multiply(rt2, hs);
    Expression ct2 = affine_transform({vars[BH2], vars[X2H2], c, vars[H2H2], ght});
    ct2 = tanh(ct2);
    Expression nwt2 = cwise_multiply(zt2, ct2);
    Expression crt2 = cwise_multiply(ft2, hs);
    in = ht[i] = crt2 + nwt2;

  }
  return ht.back();
}

void GRUCONDBuilder::copy(const RNNBuilder & rnn) {
  const GRUCONDBuilder & rnn_gru = (const GRUCONDBuilder&)rnn;
  assert(params.size() == rnn_gru.params.size());
  for(size_t i = 0; i < params.size(); ++i)
      for(size_t j = 0; j < params[i].size(); ++j)
        params[i][j] = rnn_gru.params[i][j];
}

	void GRUCONDBuilder::init_parameters(int layer,int index, const std::vector<float>& vec) {
		TensorTools::SetElements(params[layer][index].get()->values,vec);

	}

} // namespace cnn
