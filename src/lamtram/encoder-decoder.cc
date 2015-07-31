#include <lamtram/encoder-decoder.h>
#include <lamtram/vocabulary.h>
#include <lamtram/macros.h>
#include <lamtram/builder-factory.h>
#include <cnn/model.h>
#include <cnn/nodes.h>
#include <cnn/rnn.h>
#include <boost/range/irange.hpp>
#include <ctime>
#include <fstream>

using namespace std;
using namespace lamtram;

EncoderDecoder::EncoderDecoder(
                   const vector<LinearEncoderPtr> & encoders,
                   const NeuralLMPtr & decoder,
                   cnn::Model & model) : encoders_(encoders), decoder_(decoder), curr_graph_(NULL) {
    // Encoder to decoder mapping parameters
    int enc2dec_in = 0;
    for(auto & enc : encoders)
        enc2dec_in += enc->GetNumLayers() * enc->GetNumNodes();
    int enc2dec_out = decoder_->GetNumLayers() * decoder_->GetNumNodes();
    p_enc2dec_W_ = model.add_parameters({enc2dec_out, enc2dec_in});
    p_enc2dec_b_ = model.add_parameters({enc2dec_out});
}


void EncoderDecoder::NewGraph(cnn::ComputationGraph & cg) {
    for(auto & enc : encoders_)
        enc->NewGraph(cg);
    decoder_->NewGraph(cg);
    i_enc2dec_b_ = parameter(cg, p_enc2dec_b_);
    i_enc2dec_W_ = parameter(cg, p_enc2dec_W_);
    curr_graph_ = &cg;
}

std::vector<cnn::expr::Expression> EncoderDecoder::GetEncodedState(
                                    const Sentence & sent_src, cnn::ComputationGraph & cg) {
    // Perform encoding with each encoder
    vector<cnn::expr::Expression> inputs;
    for(auto & enc : encoders_) {
        enc->BuildSentGraph(sent_src, cg);
        for(auto & id : enc->GetFinalHiddenLayers())
            inputs.push_back(id);
    }
    // Perform transformation
    cnn::expr::Expression i_combined;
    assert(inputs.size() > 0);
    if(inputs.size() == 1) { i_combined = inputs[0]; }
    else                   { i_combined = concatenate(inputs); }
    cnn::expr::Expression i_decin = affine_transform({i_enc2dec_b_, i_enc2dec_W_, i_combined});
    // Perform transformation
    vector<cnn::expr::Expression> decoder_in(decoder_->GetNumLayers() * 2);
    for (int i = 0; i < decoder_->GetNumLayers(); ++i) {
      decoder_in[i] = pickrange({i_decin}, i * decoder_->GetNumNodes(), (i + 1) * decoder_->GetNumNodes());
      decoder_in[i + decoder_->GetNumLayers()] = tanh({decoder_in[i]});
    }
    return decoder_in;
}

cnn::expr::Expression EncoderDecoder::BuildSentGraph(const Sentence & sent_src, const Sentence & sent_trg,
                                                  cnn::ComputationGraph & cg, LLStats & ll) {
    if(&cg != curr_graph_)
        THROW_ERROR("Initialized computation graph and passed comptuation graph don't match."); 
    // Perform encoding with each encoder
    vector<cnn::expr::Expression> decoder_in = GetEncodedState(sent_src, cg);
    return decoder_->BuildSentGraph(sent_trg, NULL, decoder_in, cg, ll);
}


EncoderDecoder* EncoderDecoder::Read(std::istream & in, cnn::Model & model) {
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
    NeuralLMPtr decoder(NeuralLM::Read(in, model));
    return new EncoderDecoder(encoders, decoder, model);
}
void EncoderDecoder::Write(std::ostream & out) {
    out << "encdec_001 " << encoders_.size() << endl;
    for(auto & enc : encoders_) enc->Write(out);
    decoder_->Write(out);
}
