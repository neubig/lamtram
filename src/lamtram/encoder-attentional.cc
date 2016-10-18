#include <lamtram/encoder-attentional.h>
#include <lamtram/macros.h>
#include <lamtram/builder-factory.h>
#include <dynet/model.h>
#include <dynet/nodes.h>
#include <dynet/rnn.h>
#include <dynet/dict.h>
#include <boost/range/irange.hpp>
#include <boost/algorithm/string.hpp>
#include <ctime>
#include <fstream>

using namespace std;
using namespace lamtram;

template <class T>
inline std::string print_vec(const std::vector<T> & vec) {
  ostringstream oss;
  if(vec.size() > 0) oss << vec[0];
  for(size_t i = 1; i < vec.size(); ++i) oss << ' ' << vec[i];
  return oss.str();
}

ExternAttentional::ExternAttentional(const std::vector<LinearEncoderPtr> & encoders,
                   const std::string & attention_type, const std::string & attention_hist, int state_size,
                   const std::string & lex_type,
                   const DictPtr & vocab_src, const DictPtr & vocab_trg,
                   int attention_context, bool source_word_embedding_in_softmax, int source_word_embedding_in_softmax_context,
                   dynet::Model & mod)
    : ExternCalculator(0), encoders_(encoders),
      attention_context_(attention_context),source_word_embedding_in_softmax_(source_word_embedding_in_softmax),source_word_embedding_in_softmax_context_(source_word_embedding_in_softmax_context),
      attention_type_(attention_type), attention_hist_(attention_hist), hidden_size_(0), state_size_(state_size), lex_type_(lex_type) {

  for(auto & enc : encoders)
    context_size_ += enc->GetNumNodes();

  if(attention_type == "dot") {
    // No parameters for dot product
  } else if(attention_type_ == "bilin") {
    p_ehid_h_W_ = mod.add_parameters({(unsigned int)state_size_, (unsigned int)context_size_});
  } else if(attention_type_.substr(0,4) == "mlp:") {
    hidden_size_ = stoi(attention_type_.substr(4));
    if(hidden_size_ == 0) hidden_size_ = GlobalVars::layer_size;
    p_ehid_h_W_ = mod.add_parameters({(unsigned int)hidden_size_, (unsigned int)context_size_});
    p_ehid_state_W_ = mod.add_parameters({(unsigned int)hidden_size_, (unsigned int)state_size_});
    p_e_ehid_W_ = mod.add_parameters({1, (unsigned int)hidden_size_});
    use_bias_ = false;
  } else if(attention_type_.substr(0,6) == "mlp_b:") {
    hidden_size_ = stoi(attention_type_.substr(6));
    assert(hidden_size_ != 0);
    p_ehid_h_W_ = mod.add_parameters({(unsigned int)hidden_size_, (unsigned int)context_size_});
    p_ehid_state_W_ = mod.add_parameters({(unsigned int)hidden_size_, (unsigned int)state_size_});
    p_ehid_h_b_ = mod.add_parameters({(unsigned int)hidden_size_, 1});
    p_e_ehid_W_ = mod.add_parameters({1, (unsigned int)hidden_size_});
    p_e_ehid_b_ = mod.add_parameters({1, 1});
    use_bias_ = true;
  } else {
    THROW_ERROR("Illegal attention type: " << attention_type);
  }
  // Create the attention history type
  if(attention_hist == "sum") {
    p_align_sum_W_ = mod.add_parameters({1});
    p_align_sum_W_.zero();
  } else if(attention_hist != "none") {
    THROW_ERROR("Illegal attention history type: " << attention_hist);
  }

  // If we're using a lexicon, create it
  // TODO: Maybe split this into a separate class?
  if(lex_type_ != "none") {
    std::vector<string> strs;
    boost::split(strs, lex_type_, boost::is_any_of(":"));
    if(strs[0] == "prior") {
      for(size_t i = 1; i < strs.size(); ++i) {
        if(strs[i].substr(0,5) == "file=") {
          lex_file_ = strs[i].substr(5);
          lex_mapping_.reset(LoadMultipleIdMapping(lex_file_, vocab_src, vocab_trg));
        } else if(strs[i].substr(0,6) == "alpha=") {
          lex_alpha_ = stof(strs[i].substr(6));
          if(lex_alpha_ <= 0) THROW_ERROR("Value alpha for lexicon must be larger than zero");
        } else {
          THROW_ERROR("Illegal lexicon type: " << lex_type_);
        }
        assert(lex_mapping_.get() != nullptr);
      }
      // Do post-processing
      lex_size_ = vocab_trg->size();
      for(auto & lex_val : *lex_mapping_) {
        map<WordId,float> prob_map;
        for(auto & kv : lex_val.second)
          prob_map[kv.first] += kv.second;
        lex_val.second.clear();
        for(auto & kv : prob_map)
          lex_val.second.push_back(make_pair(kv.first, log(kv.second + lex_alpha_)));
      }
      lex_alpha_ = log(lex_alpha_);
    } else {
      THROW_ERROR("Illegal lexicon type: " << lex_type_);
    }
  }

}


dynet::expr::Expression ExternAttentional::CalcPrior(
                      const dynet::expr::Expression & align_vec) const {
  return (i_lexicon_.pg != nullptr ? i_lexicon_ * align_vec : dynet::expr::Expression());
}


// Index the parameters in a computation graph
void ExternAttentional::NewGraph(dynet::ComputationGraph & cg) {
  for(auto & enc : encoders_)
    enc->NewGraph(cg);
  if(attention_type_ != "dot")
    i_ehid_h_W_ = parameter(cg, p_ehid_h_W_);
  if(hidden_size_) {
    i_ehid_state_W_ = parameter(cg, p_ehid_state_W_);
    i_e_ehid_W_ = parameter(cg, p_e_ehid_W_);
    if(use_bias_) {
      i_ehid_h_b_ = parameter(cg, p_ehid_h_b_);
      i_e_ehid_b_ = parameter(cg, p_e_ehid_b_);
    }
  }
  if(attention_hist_ == "sum") {
    i_align_sum_W_ = parameter(cg, p_align_sum_W_);
  }
  curr_graph_ = &cg;
}

ExternAttentional* ExternAttentional::Read(std::istream & in, const DictPtr & vocab_src, const DictPtr & vocab_trg, dynet::Model & model) {
  int num_encoders, state_size;
  string version_id, attention_type, attention_hist = "none", lex_type = "none", line;
  int attention_context = 0;
  bool source_word_embedding_in_softmax = false;
  int source_word_embedding_in_softmax_context = 0;

  if(!getline(in, line))
    THROW_ERROR("Premature end of model file when expecting ExternAttentional");
  istringstream iss(line);
  iss >> version_id;
  if(version_id == "extatt_002") {
    iss >> num_encoders >> attention_type >> state_size;
  } else if (version_id == "extatt_003") {
    iss >> num_encoders >> attention_type >> attention_hist >> state_size;
  } else if (version_id == "extatt_004") {
    iss >> num_encoders >> attention_type >> attention_hist >> lex_type >> state_size;
  } else if (version_id == "extatt_005") {
    iss >> num_encoders >> attention_type >> attention_hist >> lex_type >> state_size >> attention_context >> source_word_embedding_in_softmax >> source_word_embedding_in_softmax_context;
  } else {
    THROW_ERROR("Expecting a ExternAttentional of version extatt_002-extatt_004, but got something different:" << endl << line);
  }
  
  vector<LinearEncoderPtr> encoders;
  while(num_encoders-- > 0)
    encoders.push_back(LinearEncoderPtr(LinearEncoder::Read(in, model)));
  return new ExternAttentional(encoders, attention_type, attention_hist, state_size, lex_type, vocab_src, vocab_trg, attention_context,source_word_embedding_in_softmax, source_word_embedding_in_softmax_context, model);
}

void ExternAttentional::Write(std::ostream & out) {
  out << "extatt_005 " << encoders_.size() << " " << attention_type_ << " " << attention_hist_ << " " << lex_type_ << " " << state_size_ << " " << attention_context_ << " " << source_word_embedding_in_softmax_ << " " << source_word_embedding_in_softmax_context_ << endl;
  for(auto & enc : encoders_) enc->Write(out);
}


void ExternAttentional::InitializeSentence(
      const Sentence & sent_src, bool train, dynet::ComputationGraph & cg) {

  // First get the states in a digestable format
  vector<vector<dynet::expr::Expression> > hs_sep;
  vector<vector<dynet::expr::Expression> > wes_sep;
  for(auto & enc : encoders_) {
    enc->BuildSentGraph(sent_src, true, train, cg);
    hs_sep.push_back(enc->GetWordStates());
    wes_sep.push_back(enc->GetWordEmbeddings());
    assert(hs_sep[0].size() == hs_sep.rbegin()->size());
  }
  sent_len_ = hs_sep[0].size();
  
  // Concatenate them if necessary
  vector<dynet::expr::Expression> hs_comb;
  if(encoders_.size() == 1) {
    hs_comb = hs_sep[0];
  } else {
    for(int i : boost::irange(0, sent_len_)) {
      vector<dynet::expr::Expression> vars;
      for(int j : boost::irange(0, (int)encoders_.size())) {
        vars.push_back(hs_sep[j][i]);
      }
      hs_comb.push_back(concatenate(vars));
    }
  }
  i_h_ = concatenate_cols(hs_comb);
  i_h_last_ = *hs_comb.rbegin();

  //vector for attention context
  vector<dynet::expr::Expression> hcs_comb;
  if(attention_context_ > 0 ) {
    for(int i : boost::irange(0, sent_len_)) {
      vector<dynet::expr::Expression> vars;
      for(int k : boost::irange(1,attention_context_+1)) {
        for(int j : boost::irange(0, (int)encoders_.size())) {
          if(i+k < 0 || i+k >= sent_len_) {
            vars.push_back(zeroes(cg,hs_sep[j][i].value().d));
          }else {
            vars.push_back(hs_sep[j][i+k]);
          }
          if(i-k < 0 || i-k >= sent_len_) {
            vars.push_back(zeroes(cg,hs_sep[j][i].value().d));
          }else {
            vars.push_back(hs_sep[j][i-k]);
          }
        }
      }
      hcs_comb.push_back(concatenate(vars));
    }
    i_hc_ = concatenate_cols(hcs_comb);
  }
  
  //vector for source word embedding with context  
  vector<dynet::expr::Expression> wes_comb;
  if(source_word_embedding_in_softmax_) {
    if(encoders_.size() == 1 && source_word_embedding_in_softmax_context_ == 0) {
      wes_comb = hs_sep[0];
    }else {
      for(int i : boost::irange(0, sent_len_)) {
        vector<dynet::expr::Expression> vars;
        for(int j : boost::irange(0, (int)encoders_.size())) {
          vars.push_back(wes_sep[j][i]);
        }
        for(int k : boost::irange(1,source_word_embedding_in_softmax_context_+1)) {
          for(int j : boost::irange(0, (int)encoders_.size())) {
            if(i+k < 0 || i+k >= sent_len_) {
              vars.push_back(zeroes(cg,wes_sep[j][i].value().d));
            }else {
              vars.push_back(wes_sep[j][i+k]);
            }
            if(i-k < 0 || i-k >= sent_len_) {
              vars.push_back(zeroes(cg,wes_sep[j][i].value().d));
            }else {
              vars.push_back(wes_sep[j][i-k]);
            }
          }
        }
        wes_comb.push_back(concatenate(vars));
      }
    }
    i_we_ = concatenate_cols(wes_comb);
  }


  // Create an identity with shape
  if(hidden_size_) {
    if(use_bias_) {
      i_ehid_hpart_ = i_ehid_h_W_ *  i_h_;
      i_ehid_hpart_ = colwise_add(i_ehid_hpart_,i_ehid_h_b_);
    }else {
      i_ehid_hpart_ = i_ehid_h_W_*i_h_;
    }
    sent_values_.resize(sent_len_, 1.0);
    i_sent_len_ = input(cg, {1, (unsigned int)sent_len_}, &sent_values_);
  } else if(attention_type_ == "dot") {
    i_ehid_hpart_ = transpose(i_h_);
  } else if(attention_type_ == "bilin") {
    i_ehid_hpart_ = transpose(i_ehid_h_W_*i_h_);
  } else {
    THROW_ERROR("Bad attention type " << attention_type_);
  }

  // If we're using a lexicon, create the values
  if(lex_type_ != "none") {
    vector<float> lex_data;
    vector<unsigned int> lex_ids;
    unsigned int start = 0;
    for(size_t i = 0; i < sent_len_; ++i, start += lex_size_) {
      WordId wid = (i < sent_src.size() ? sent_src[i] : 0);
      auto it = lex_mapping_->find(wid);
      if(it != lex_mapping_->end()) {
        for(auto & kv : it->second) {
          lex_ids.push_back(start + kv.first);
          lex_data.push_back(kv.second);
        }
      }
    }
    i_lexicon_ = input(cg, {(unsigned int)lex_size_, (unsigned int)sent_len_}, lex_ids, lex_data, lex_alpha_);
  }

}

void ExternAttentional::InitializeSentence(
      const std::vector<Sentence> & sent_src, bool train, dynet::ComputationGraph & cg) {

  // First get the states in a digestable format
  vector<vector<dynet::expr::Expression> > hs_sep;
  vector<vector<dynet::expr::Expression> > wes_sep;
  for(auto & enc : encoders_) {
    enc->BuildSentGraph(sent_src, true, train, cg);
    hs_sep.push_back(enc->GetWordStates());
    wes_sep.push_back(enc->GetWordEmbeddings());
    assert(hs_sep[0].size() == hs_sep.rbegin()->size());
  }
  sent_len_ = hs_sep[0].size();
  // Concatenate them if necessary
  vector<dynet::expr::Expression> hs_comb;
  if(encoders_.size() == 1) {
    hs_comb = hs_sep[0];
  } else {
    for(int i : boost::irange(0, sent_len_)) {
      vector<dynet::expr::Expression> vars;
      for(int j : boost::irange(0, (int)encoders_.size())) {
        vars.push_back(hs_sep[j][i]);
      }
      hs_comb.push_back(concatenate(vars));
    }
  }
  i_h_ = concatenate_cols(hs_comb);
  i_h_last_ = *hs_comb.rbegin();


  //vector for attention context
  vector<dynet::expr::Expression> hcs_comb;
  if(attention_context_ > 0 ) {
    for(int i : boost::irange(0, sent_len_)) {
      vector<dynet::expr::Expression> vars;
      for(int k : boost::irange(1,attention_context_+1)) {
        for(int j : boost::irange(0, (int)encoders_.size())) {
          if(i+k < 0 || i+k >= sent_len_) {
            vars.push_back(zeroes(cg,hs_sep[j][i].value().d));
          }else {
            vars.push_back(hs_sep[j][i+k]);
          }
          if(i-k < 0 || i-k >= sent_len_) {
            vars.push_back(zeroes(cg,hs_sep[j][i].value().d));
          }else {
            vars.push_back(hs_sep[j][i-k]);
          }
        }
      }
      hcs_comb.push_back(concatenate(vars));
    }
    i_hc_ = concatenate_cols(hcs_comb);
  }


  //vector for source word embedding with context  
  vector<dynet::expr::Expression> wes_comb;
  if(source_word_embedding_in_softmax_) {
    if(encoders_.size() == 1 && source_word_embedding_in_softmax_context_ == 0) {
      wes_comb = hs_sep[0];
    }else {
      for(int i : boost::irange(0, sent_len_)) {
        vector<dynet::expr::Expression> vars;
        for(int j : boost::irange(0, (int)encoders_.size())) {
          vars.push_back(wes_sep[j][i]);
        }
        for(int k : boost::irange(1,source_word_embedding_in_softmax_context_+1)) {
          for(int j : boost::irange(0, (int)encoders_.size())) {
            if(i+k < 0 || i+k >= sent_len_) {
              vars.push_back(zeroes(cg,wes_sep[j][i].value().d));
            }else {
              vars.push_back(wes_sep[j][i+k]);
            }
            if(i-k < 0 || i-k >= sent_len_) {
              vars.push_back(zeroes(cg,wes_sep[j][i].value().d));
            }else {
              vars.push_back(wes_sep[j][i-k]);
            }
          }
        }
        wes_comb.push_back(concatenate(vars));
      }
    }
    i_we_ = concatenate_cols(wes_comb);
  }


  // Create an identity with shape
  if(hidden_size_) {
    if(use_bias_) {
      i_ehid_hpart_ = i_ehid_h_W_ *  i_h_;
      i_ehid_hpart_ = colwise_add(i_ehid_hpart_,i_ehid_h_b_);
    }else {
      i_ehid_hpart_ = i_ehid_h_W_*i_h_;
    }
    sent_values_.resize(sent_len_, 1.0);
    i_sent_len_ = input(cg, {1, (unsigned int)sent_len_}, &sent_values_);
  } else if(attention_type_ == "dot") {
    i_ehid_hpart_ = transpose(i_h_);
  } else if(attention_type_ == "bilin") {
    i_ehid_hpart_ = transpose(i_ehid_h_W_*i_h_);
  } else {
    THROW_ERROR("Bad attention type " << attention_type_);
  }

  // If we're using a lexicon, create it
  if(lex_type_ != "none") {
    vector<float> lex_data;
    vector<unsigned int> lex_ids;
    unsigned int start = 0;
    for(size_t j = 0; j < sent_src.size(); ++j) {
      for(size_t i = 0; i < sent_len_; ++i, start += lex_size_) {
        if(i > sent_src[j].size()) continue;
        WordId wid = (i < sent_src[j].size() ? sent_src[j][i] : 0);
        auto it = lex_mapping_->find(wid);
        if(it != lex_mapping_->end()) {
          for(auto & kv : it->second) {
            lex_ids.push_back(start + kv.first);
            lex_data.push_back(kv.second);
          }
        }
      }
    }
    i_lexicon_ = input(cg, dynet::Dim({(unsigned int)lex_size_, (unsigned int)sent_len_}, (unsigned int)sent_src.size()), lex_ids, lex_data, lex_alpha_);
  }

}

dynet::expr::Expression ExternAttentional::GetEmptyContext(dynet::ComputationGraph & cg) const {
  return zeroes(cg, {(unsigned int)state_size_});
}

// Create a variable encoding the context
dynet::expr::Expression ExternAttentional::CreateContext(
    const std::vector<dynet::expr::Expression> & state_in,
    const dynet::expr::Expression & align_sum_in,
    bool train,
    dynet::ComputationGraph & cg,
    std::vector<dynet::expr::Expression> & align_out,
    dynet::expr::Expression & align_sum_out) const {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match."); 
  dynet::expr::Expression i_ehid, i_e;
  // MLP
  if(hidden_size_) {
    if(state_in.size()) {
      // i_ehid_state_W_ is {hidden_size, state_size}, state_in is {state_size, 1}
      dynet::expr::Expression i_ehid_spart = i_ehid_state_W_ * *state_in.rbegin();
      i_ehid = affine_transform({i_ehid_hpart_, i_ehid_spart, i_sent_len_});
    } else {
      i_ehid = i_ehid_hpart_;
    }
    // Run through nonlinearity
    dynet::expr::Expression i_ehid_out = tanh({i_ehid});
    // i_e_ehid_W_ is {1, hidden_size}, i_ehid_out is {hidden_size, sent_len}
    if(use_bias_) {
      i_e = transpose(colwise_add(i_e_ehid_W_ * i_ehid_out,i_e_ehid_b_));
    }else {
      i_e = transpose(i_e_ehid_W_ * i_ehid_out);
    }
  // Bilinear/dot product
  } else {
    assert(state_in.size() > 0);
    i_e = i_ehid_hpart_ * (*state_in.rbegin());
  }
  dynet::expr::Expression i_alpha;
  // Calculate the softmax, adding the previous sum if necessary
  if(align_sum_in.pg != nullptr) {
    i_alpha = softmax(i_e + align_sum_in * i_align_sum_W_);
    // // DEBUG
    // dynet::Tensor align_sum_tens = align_sum_in.value();
    // vector<float> align_sum_val = as_vector(align_sum_in.value());
  } else {
    i_alpha = softmax(i_e);
  }
  // Save the alignments and print if necessary
  align_out.push_back(i_alpha);
  if(GlobalVars::verbose >= 2) {
    vector<dynet::real> softmax = as_vector(cg.incremental_forward(i_alpha));
    cerr << "Alignments: " << softmax << endl;
  }
  // Update the sum if necessary
  if(attention_hist_ == "sum") {
    align_sum_out = (align_sum_in.pg != nullptr ? align_sum_in + i_alpha : i_alpha);
  }
  // i_h_ is {input_size, sent_len}, i_alpha is {sent_len, 1}
  return i_h_ * i_alpha; 
}


dynet::expr::Expression ExternAttentional::CalcAttentionContext(const dynet::expr::Expression align) const {

  return i_hc_ * align;

}

dynet::expr::Expression ExternAttentional::CalcWordContext(const dynet::expr::Expression align) const {
  return i_we_ * align;
}




EncoderAttentional::EncoderAttentional(
           const ExternAttentionalPtr & extern_calc,
           const NeuralLMPtr & decoder,
           dynet::Model & model)
  : extern_calc_(extern_calc), decoder_(decoder), curr_graph_(NULL) {
  // Encoder to decoder mapping parameters
  int enc2dec_in = extern_calc->GetContextSize();
  int enc2dec_out = decoder_->GetNumLayers() * decoder_->GetNumNodes();
  p_enc2dec_W_ = model.add_parameters({(unsigned int)enc2dec_out, (unsigned int)enc2dec_in});
  p_enc2dec_b_ = model.add_parameters({(unsigned int)enc2dec_out});
}


void EncoderAttentional::NewGraph(dynet::ComputationGraph & cg) {
  extern_calc_->NewGraph(cg);
  decoder_->NewGraph(cg);
  i_enc2dec_b_ = parameter(cg, p_enc2dec_b_);
  i_enc2dec_W_ = parameter(cg, p_enc2dec_W_);
  curr_graph_ = &cg;
}

template <class SentData>
std::vector<dynet::expr::Expression> EncoderAttentional::GetEncodedState(const SentData & sent_src, bool train, dynet::ComputationGraph & cg) {
  extern_calc_->InitializeSentence(sent_src, train, cg);
  dynet::expr::Expression i_decin = affine_transform({i_enc2dec_b_, i_enc2dec_W_, extern_calc_->GetState()});
  // Perform transformation
  vector<dynet::expr::Expression> decoder_in(decoder_->GetNumLayers() * decoder_->GetLayerMultiplier());
  for (int i = 0; i < decoder_->GetNumLayers(); ++i) {
    decoder_in[i] = (decoder_->GetNumLayers() == 1 ?
                     i_decin :
                     pickrange({i_decin}, i * decoder_->GetNumNodes(), (i + 1) * decoder_->GetNumNodes()));
    if(decoder_->GetLayerMultiplier() == 2) {
      decoder_in[i + decoder_->GetNumLayers()] = tanh({decoder_in[i]});
    } else {
      decoder_in[i] = tanh(decoder_in[i]);
    }
  }
  return decoder_in;
}

template
std::vector<dynet::expr::Expression> EncoderAttentional::GetEncodedState<Sentence>(const Sentence & sent_src, bool train, dynet::ComputationGraph & cg);
template
std::vector<dynet::expr::Expression> EncoderAttentional::GetEncodedState<std::vector<Sentence> >(const std::vector<Sentence> & sent_src, bool train, dynet::ComputationGraph & cg);

dynet::expr::Expression EncoderAttentional::BuildSentGraph(const Sentence & sent_src,
                                                     const Sentence & sent_trg,
                                                     const Sentence & cache_trg,
                                                     const float * weight,
                                                     float samp_percent,
                                                     bool train,
                                                     dynet::ComputationGraph & cg,
                                                     LLStats & ll) {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
  // Perform encoding with each encoder
  vector<dynet::expr::Expression> decoder_in = GetEncodedState(sent_src, train, cg);
  return decoder_->BuildSentGraph(sent_trg, cache_trg, weight, extern_calc_.get(), decoder_in, samp_percent, train, cg, ll);
}

dynet::expr::Expression EncoderAttentional::BuildSentGraph(const std::vector<Sentence> & sent_src,
                                                     const std::vector<Sentence> & sent_trg,
                                                     const std::vector<Sentence> & cache_trg,
                                                     const std::vector<float> * weights,
                                                     float samp_percent,
                                                     bool train,
                                                     dynet::ComputationGraph & cg,
                                                     LLStats & ll) {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
  // Perform encoding with each encoder
  vector<dynet::expr::Expression> decoder_in = GetEncodedState(sent_src, train, cg);
  return decoder_->BuildSentGraph(sent_trg, cache_trg, weights, extern_calc_.get(), decoder_in, samp_percent, train, cg, ll);
}

dynet::expr::Expression EncoderAttentional::SampleTrgSentences(const Sentence & sent_src,
                                                             const Sentence * sent_trg,
                                                             int num_samples,
                                                             int max_len,
                                                             bool train,
                                                             dynet::ComputationGraph & cg,
                                                             vector<Sentence> & samples) {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match."); 
  std::vector<dynet::expr::Expression> decoder_in = GetEncodedState(sent_src, train, cg);
  return decoder_->SampleTrgSentences(extern_calc_.get(), decoder_in, sent_trg, num_samples, max_len, train, cg, samples);
}

EncoderAttentional* EncoderAttentional::Read(const DictPtr & vocab_src, const DictPtr & vocab_trg, std::istream & in, dynet::Model & model) {
  string version_id, line;
  if(!getline(in, line))
    THROW_ERROR("Premature end of model file when expecting EncoderAttentional");
  istringstream iss(line);
  iss >> version_id;
  if(version_id != "encatt_001")
    THROW_ERROR("Expecting a EncoderAttentional of version encatt_001, but got something different:" << endl << line);
  ExternAttentionalPtr extern_calc(ExternAttentional::Read(in, vocab_src, vocab_trg, model));
  NeuralLMPtr decoder(NeuralLM::Read(vocab_trg, in, model));
  decoder->SetAttention(extern_calc);
  return new EncoderAttentional(extern_calc, decoder, model);
}
EncoderAttentional* EncoderAttentional::Convert(const DictPtr & vocab_src, const DictPtr & vocab_trg, const std::string & file, const boost::property_tree::ptree & json, dynet::Model & model) {
  
  
  vector<LinearEncoderPtr> encoders;

  int wordrep= json.get_child("dim_word").get_value<int>();
  int dim= json.get_child("dim").get_value<int>();
  std::stringstream s;
  s << "gru:" << dim << ":1";

  BuilderSpec enc_layer_spec(s.str());
  s.str(std::string());

  s << "mlp_b:" << 2*dim;

  std::string attention_type = s.str();
  s.str(std::string());
  std::string attention_hist = "none";
  std::string attention_lex = "none";
  bool attention_feed = false;
  //bool attention_feed = true;
  //int context=2;
  int context=1;
  
  s << "multilayer:" << wordrep << ":full";
  std:string softmax_sig = s.str();
  s.str(std::string());

  LinearEncoderPtr enc(new LinearEncoder(vocab_src->size(), wordrep, enc_layer_spec, vocab_src->get_unk_id(), model));
  encoders.push_back(enc);
  LinearEncoderPtr revenc(new LinearEncoder(vocab_src->size(), wordrep, enc_layer_spec, vocab_src->get_unk_id(), model));
  cout << "Source voc size:" << vocab_src->size() << endl;
  revenc->SetReverse(true);
  encoders.push_back(revenc);

  //BuilderSpec dec_layer_spec(enc_layer_spec); dec_layer_spec.nodes *= encoders.size();
  s << "gru-cond:" << dim << ":1";
  BuilderSpec dec_layer_spec(s.str());
  s.str(std::string());

  int ctxdim = 2*dim;

  ExternAttentionalPtr extatt(new ExternAttentional(encoders, attention_type, attention_hist, dim, attention_lex, vocab_src, vocab_trg, 0,false, 0,model));
  
  ExternCalculatorPtr p = extatt;
  NeuralLMPtr decoder(new NeuralLM(vocab_trg, context, ctxdim, attention_feed, wordrep, dec_layer_spec, vocab_trg->get_unk_id(), softmax_sig, true, 0, false, 0, p, model));
  EncoderAttentional * encatt = new EncoderAttentional(extatt, decoder, model);
  
  //convert weights
  
  cout << "Loading npz file: \"" << file << "\"" << endl;
  cnpy::npz_t npz_model(cnpy::npz_load(file));


  bool dropOut = json.get_child("use_dropout").get_value<bool>();
  float dropoutProb = 1;
  if(dropOut) {
    dropoutProb = 1 - json.get_child("dropout_source").get_value<float>();
  }

  CnpyUtils::copyWeight("Wemb",npz_model,enc->p_wr_W_,dropoutProb);
  CnpyUtils::copyWeight("Wemb",npz_model,revenc->p_wr_W_,dropoutProb);
  
  CnpyUtils::copyWeight("Wemb_dec",npz_model,decoder->p_wr_W_,1);

  CnpyUtils::copyGRUWeight("encoder_",npz_model,enc->builder_);
  CnpyUtils::copyGRUWeight("encoder_r_",npz_model,revenc->builder_);

  CnpyUtils::copyGRUCondWeight("decoder_",npz_model,decoder->builder_);
  CnpyUtils::copyAttentionWeight("decoder_",npz_model,extatt);
  
  CnpyUtils::copySoftmaxWeight("ff_",npz_model,decoder->softmax_,vocab_trg->size());
  
  //for(auto const &e : npz_model) {
    //cout << e.first << endl;
    //cout << e.second.shape[0] << " " << e.second.shape[1] << endl;
  //}
  
  cout << "Done" << endl;
  return encatt;
}



void EncoderAttentional::Write(std::ostream & out) {
  out << "encatt_001" << endl;
  extern_calc_->Write(out);
  decoder_->Write(out);
}
