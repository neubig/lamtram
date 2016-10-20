#include "lamtram/lstm-cond.h"

#include <fstream>
#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "dynet/nodes.h"
#include "dynet/io-macros.h"

using namespace std;
using namespace dynet::expr;

namespace lamtram {

enum { X2I, H2I, C2I, BI, X2O, H2O, C2O, BO, X2C, H2C, BC ,X2I2, H2I2, C2I2, BI2, X2O2, H2O2, C2O2, BO2, X2C2, H2C2, BC2 };

LSTMCONDBuilder::LSTMCONDBuilder(unsigned layers,
                         unsigned input_dim,
                         unsigned input_2_dim,
                         unsigned hidden_dim,
                         Model* model,ExternCalculatorPtr & att) : layers(layers) {
  att_ = att;
  unsigned layer_input_dim = input_dim;
  unsigned layer_input_2_dim = input_2_dim;

  for (unsigned i = 0; i < layers; ++i) {
    // i
    Parameter p_x2i = model->add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2i = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_c2i = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bi = model->add_parameters({hidden_dim});

    // o
    Parameter p_x2o = model->add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2o = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_c2o = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bo = model->add_parameters({hidden_dim});

    // c
    Parameter p_x2c = model->add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2c = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bc = model->add_parameters({hidden_dim});
    
    //and the second step after attention
    // i
    Parameter p_x2i_2 = model->add_parameters({hidden_dim, layer_input_2_dim});
    Parameter p_h2i_2 = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_c2i_2 = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bi_2 = model->add_parameters({hidden_dim});

    // o
    Parameter p_x2o_2 = model->add_parameters({hidden_dim, layer_input_2_dim});
    Parameter p_h2o_2 = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_c2o_2 = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bo_2 = model->add_parameters({hidden_dim});

    // c
    Parameter p_x2c_2 = model->add_parameters({hidden_dim, layer_input_2_dim});
    Parameter p_h2c_2 = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bc_2 = model->add_parameters({hidden_dim});

    
    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameter> ps = {p_x2i, p_h2i, p_c2i, p_bi, p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc,
                            p_x2i_2, p_h2i_2, p_c2i_2, p_bi_2, p_x2o_2, p_h2o_2, p_c2o_2, p_bo_2, p_x2c_2, p_h2c_2, p_bc_2};
    params.push_back(ps);
  }  // layers
  dropout_rate = 0.f;  
}

void LSTMCONDBuilder::new_graph_impl(ComputationGraph& cg){
  param_vars.clear();

  for (unsigned i = 0; i < layers; ++i){
    auto& p = params[i];

    //i
    Expression i_x2i = parameter(cg,p[X2I]);
    Expression i_h2i = parameter(cg,p[H2I]);
    Expression i_c2i = parameter(cg,p[C2I]);
    Expression i_bi = parameter(cg,p[BI]);
    //o
    Expression i_x2o = parameter(cg,p[X2O]);
    Expression i_h2o = parameter(cg,p[H2O]);
    Expression i_c2o = parameter(cg,p[C2O]);
    Expression i_bo = parameter(cg,p[BO]);
    //c
    Expression i_x2c = parameter(cg,p[X2C]);
    Expression i_h2c = parameter(cg,p[H2C]);
    Expression i_bc = parameter(cg,p[BC]);

    //and the second step after attention
    //i
    Expression i_x2i_2 = parameter(cg,p[X2I2]);
    Expression i_h2i_2 = parameter(cg,p[H2I2]);
    Expression i_c2i_2 = parameter(cg,p[C2I2]);
    Expression i_bi_2 = parameter(cg,p[BI2]);
    //o
    Expression i_x2o_2 = parameter(cg,p[X2O2]);
    Expression i_h2o_2 = parameter(cg,p[H2O2]);
    Expression i_c2o_2 = parameter(cg,p[C2O2]);
    Expression i_bo_2 = parameter(cg,p[BO2]);
    //c
    Expression i_x2c_2 = parameter(cg,p[X2C2]);
    Expression i_h2c_2 = parameter(cg,p[H2C2]);
    Expression i_bc_2 = parameter(cg,p[BC2]);

    vector<Expression> vars = {i_x2i, i_h2i, i_c2i, i_bi, i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc,
                                i_x2i_2, i_h2i_2, i_c2i_2, i_bi_2, i_x2o_2, i_h2o_2, i_c2o_2, i_bo_2, i_x2c_2, i_h2c_2, i_bc_2};
    param_vars.push_back(vars);
  }
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void LSTMCONDBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();
  if (hinit.size() > 0) {
    assert(layers*2 == hinit.size());
    h0.resize(layers);
    c0.resize(layers);
    for (unsigned i = 0; i < layers; ++i) {
      c0[i] = hinit[i];
      h0[i] = hinit[i + layers];
    }
    has_initial_state = true;
  } else {
    has_initial_state = false;
  }
}

// TO DO - Make this correct
Expression LSTMCONDBuilder::set_h_impl(int prev, const vector<Expression>& h_new) {
  if (h_new.size()) { assert(h_new.size() == layers); }
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    Expression y = h_new[i];
    h[t][i] = y;
  }
  return h[t].back();
}

Expression LSTMCONDBuilder::add_input_impl(int prev, const Expression& x) {
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();
  Expression in = x;
  for (unsigned i = 0; i < layers; ++i) {

    const vector<Expression>& vars = param_vars[i];
    Expression i_h_tm1, i_c_tm1;
    bool has_prev_state = (prev >= 0 || has_initial_state);
    if (prev < 0) {
      if (has_initial_state) {
        // intial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_tm1 = h0[i];
        i_c_tm1 = c0[i];
      }
    } else {  // t > 0
      i_h_tm1 = h[prev][i];
      i_c_tm1 = c[prev][i];
    }
    // apply dropout according to http://arxiv.org/pdf/1409.2329v5.pdf
    if (dropout_rate) in = dropout(in, dropout_rate);

    // input
    Expression i_ait;
    if (has_prev_state)
      i_ait = affine_transform({vars[BI], vars[X2I], in, vars[H2I], i_h_tm1, vars[C2I], i_c_tm1});
    else
      i_ait = affine_transform({vars[BI], vars[X2I], in});
    Expression i_it = logistic(i_ait);
    // forget
   Expression i_ft = 1.f - i_it;
    // write memory cell
    Expression i_awt;
    if (has_prev_state)
      i_awt = affine_transform({vars[BC], vars[X2C], in, vars[H2C], i_h_tm1});
    else
      i_awt = affine_transform({vars[BC], vars[X2C], in});
    Expression i_wt = tanh(i_awt);
    // output
    Expression cs;
    if (has_prev_state) {
      Expression i_nwt = cwise_multiply(i_it,i_wt);
      Expression i_crt = cwise_multiply(i_ft,i_c_tm1);
      cs = i_crt + i_nwt;
    } else {
      cs = cwise_multiply(i_it,i_wt);
    }

    Expression i_aot;
    if (has_prev_state)
      i_aot = affine_transform({vars[BO], vars[X2O], in, vars[H2O], i_h_tm1, vars[C2O], cs});
    else
      i_aot = affine_transform({vars[BO], vars[X2O], in, vars[C2O], cs});
    Expression i_ot = logistic(i_aot);
    Expression ph_t = tanh(cs);
    Expression hs = cwise_multiply(i_ot,ph_t);


    //get context;
    vector<Expression> hs2;
    hs2.push_back(hs);
    attention_context_ = att_->CreateContext(hs2,*align_sum_in_,train_,*cg_,*align_out_,*align_sum_out_);
    Expression att_context = attention_context_;
    if (dropout_rate) att_context = dropout(att_context,dropout_rate);

    // input
    i_ait = affine_transform({vars[BI2], vars[X2I2], att_context, vars[H2I2], hs, vars[C2I2], cs});
    i_it = logistic(i_ait);
    // forget
    i_ft = 1.f - i_it;
    // write memory cell
    i_awt = affine_transform({vars[BC2], vars[X2C2], att_context, vars[H2C2], hs});
    i_wt = tanh(i_awt);
    // output
    Expression i_nwt = cwise_multiply(i_it,i_wt);
    Expression i_crt = cwise_multiply(i_ft,cs);
    ct[i] = i_crt + i_nwt;

    i_aot = affine_transform({vars[BO2], vars[X2O2], att_context, vars[H2O2], hs, vars[C2O2], ct[i]});
    i_ot = logistic(i_aot);
    ph_t = tanh(ct[i]);
    in = ht[i] = cwise_multiply(i_ot,ph_t);

  }

  
  if (dropout_rate) return dropout(ht.back(), dropout_rate);
    else return ht.back();
}

void LSTMCONDBuilder::copy(const RNNBuilder & rnn) {
  const LSTMCONDBuilder & rnn_lstm = (const LSTMCONDBuilder&)rnn;
  assert(params.size() == rnn_lstm.params.size());
  for(size_t i = 0; i < params.size(); ++i)
      for(size_t j = 0; j < params[i].size(); ++j)
        params[i][j] = rnn_lstm.params[i][j];
}

void LSTMCONDBuilder::save_parameters_pretraining(const string& fname) const {
  cerr << "Writing LSTMCOND parameters to " << fname << endl;
  ofstream of(fname);
  assert(of);
  boost::archive::binary_oarchive oa(of);
  std::string id = "LSTMCONDBuilder:params";
  oa << id;
  oa << layers;
  for (unsigned i = 0; i < layers; ++i) {
    for (auto p : params[i]) {
      oa << p.get()->values;
    }
  }
}

void LSTMCONDBuilder::load_parameters_pretraining(const string& fname) {
  cerr << "Loading LSTMCOND parameters from " << fname << endl;
  ifstream of(fname);
  assert(of);
  boost::archive::binary_iarchive ia(of);
  std::string id;
  ia >> id;
  if (id != "LSTMCONDBuilder:params") {
    cerr << "Bad id read\n";
    abort();
  }
  unsigned l = 0;
  ia >> l;
  if (l != layers) {
    cerr << "Bad number of layers\n";
    abort();
  }
  // TODO check other dimensions
  for (unsigned i = 0; i < layers; ++i) {
    for (auto p : params[i]) {
      ia >> p.get()->values;
    }
  }
}

template<class Archive>
void LSTMCONDBuilder::serialize(Archive& ar, const unsigned int) {
  ar & boost::serialization::base_object<RNNBuilder>(*this);
  ar & params;
  ar & layers;
  ar & dropout_rate;
}
DYNET_SERIALIZE_IMPL(LSTMCONDBuilder);

} // namespace dynet
