#include <lamtram/encoder-decoder.h>
#include <lamtram/macros.h>
#include <lamtram/builder-factory.h>
#include <dynet/model.h>
#include <dynet/nodes.h>
#include <dynet/rnn.h>
#include <boost/range/irange.hpp>
#include <ctime>
#include <fstream>

using namespace std;
using namespace lamtram;

EncoderDecoder::EncoderDecoder(
           const vector<LinearEncoderPtr> & encoders,
           const NeuralLMPtr & decoder,
           dynet::Model & model) : encoders_(encoders), decoder_(decoder), curr_graph_(NULL) {
  // Encoder to decoder mapping parameters
  int enc2dec_in = 0;
  for(auto & enc : encoders)
    enc2dec_in += enc->GetNumLayers() * enc->GetNumNodes();
  int enc2dec_out = decoder_->GetNumLayers() * decoder_->GetNumNodes();
  p_enc2dec_W_ = model.add_parameters({(unsigned int)enc2dec_out, (unsigned int)enc2dec_in});
  p_enc2dec_b_ = model.add_parameters({(unsigned int)enc2dec_out});
}


void EncoderDecoder::NewGraph(dynet::ComputationGraph & cg) {
  for(auto & enc : encoders_)
    enc->NewGraph(cg);
  decoder_->NewGraph(cg);
  i_enc2dec_b_ = parameter(cg, p_enc2dec_b_);
  i_enc2dec_W_ = parameter(cg, p_enc2dec_W_);
  curr_graph_ = &cg;
}

template <class SentData>
std::vector<dynet::expr::Expression> EncoderDecoder::GetEncodedState(
                  const SentData & sent_src, bool train, dynet::ComputationGraph & cg) {
  // Perform encoding with each encoder
  vector<dynet::expr::Expression> inputs;
  for(auto & enc : encoders_) {
    enc->BuildSentGraph(sent_src, true, train, cg);
    for(auto & id : enc->GetFinalHiddenLayers())
      inputs.push_back(id);
  }
  // Perform transformation
  dynet::expr::Expression i_combined;
  assert(inputs.size() > 0);
  if(inputs.size() == 1) { i_combined = inputs[0]; }
  else           { i_combined = concatenate(inputs); }
  dynet::expr::Expression i_decin = affine_transform({i_enc2dec_b_, i_enc2dec_W_, i_combined});
  // Perform transformation
  vector<dynet::expr::Expression> decoder_in(decoder_->GetNumLayers() * decoder_->GetLayerMultiplier());
  for (int i = 0; i < decoder_->GetNumLayers(); ++i) {
    decoder_in[i] = pickrange({i_decin}, i * decoder_->GetNumNodes(), (i + 1) * decoder_->GetNumNodes());
    if(decoder_->GetLayerMultiplier() == 2) {
      decoder_in[i + decoder_->GetNumLayers()] = tanh({decoder_in[i]});
    } else {
      decoder_in[i] = tanh(decoder_in[i]);
    }
  }
  return decoder_in;
}

template
std::vector<dynet::expr::Expression> EncoderDecoder::GetEncodedState<Sentence>(
                  const Sentence & sent_src, bool train, dynet::ComputationGraph & cg);
template
std::vector<dynet::expr::Expression> EncoderDecoder::GetEncodedState<vector<Sentence> >(
                  const vector<Sentence> & sent_src, bool train, dynet::ComputationGraph & cg);

dynet::expr::Expression EncoderDecoder::BuildSentGraph(const Sentence & sent_src,
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
  return decoder_->BuildSentGraph(sent_trg, cache_trg, weight, NULL, decoder_in, samp_percent, train, cg, ll);
}

dynet::expr::Expression EncoderDecoder::BuildSentGraph(const std::vector<Sentence> & sent_src,
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
  return decoder_->BuildSentGraph(sent_trg, cache_trg, weights, NULL, decoder_in, samp_percent, train, cg, ll);
}

dynet::expr::Expression EncoderDecoder::SampleTrgSentences(const Sentence & sent_src,
                                                         const Sentence * sent_trg,   
                                                         int num_samples,
                                                         int max_len,
                                                         bool train,
                                                         dynet::ComputationGraph & cg,
                                                         vector<Sentence> & samples) {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match."); 
  // Perform encoding with each encoder
  vector<dynet::expr::Expression> decoder_in = GetEncodedState(sent_src, train, cg);
  return decoder_->SampleTrgSentences(NULL, decoder_in, sent_trg, num_samples, max_len, train, cg, samples);
}

EncoderDecoder* EncoderDecoder::Read(const DictPtr & vocab_src, const DictPtr & vocab_trg, std::istream & in, dynet::Model & model) {
  int num_encoders;
  string version_id, line;
  if(!getline(in, line))
    THROW_ERROR("Premature end of model file when expecting EncoderDecoder");
  istringstream iss(line);
  iss >> version_id >> num_encoders;
  if(version_id != "encdec_001")
    THROW_ERROR("Expecting a EncoderDecoder of version encdec_001, but got something different:" << endl << line);
  vector<LinearEncoderPtr> encoders;
  while(num_encoders-- > 0)
    encoders.push_back(LinearEncoderPtr(LinearEncoder::Read(in, model)));
  NeuralLMPtr decoder(NeuralLM::Read(vocab_trg, in, model));
  return new EncoderDecoder(encoders, decoder, model);
}
void EncoderDecoder::Write(std::ostream & out) {
  out << "encdec_001 " << encoders_.size() << endl;
  for(auto & enc : encoders_) enc->Write(out);
  decoder_->Write(out);
}
