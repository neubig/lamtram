#include <lamtram/encoder-attentional.h>
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


ExternAttentional::ExternAttentional(const std::vector<LinearEncoderPtr> & encoders,
                                     int hidden_size, int state_size,
                                     cnn::Model & mod)
        : ExternCalculator(0), encoders_(encoders),
          hidden_size_(hidden_size), state_size_(state_size) {
    for(auto & enc : encoders)
        context_size_ += enc->GetNumNodes();

    p_ehid_h_W_ = mod.add_parameters({(unsigned int)hidden_size_, (unsigned int)context_size_});
    p_ehid_state_W_ = mod.add_parameters({(unsigned int)hidden_size_, (unsigned int)state_size_});
    p_e_ehid_W_ = mod.add_parameters({1, (unsigned int)hidden_size_});
}


// Index the parameters in a computation graph
void ExternAttentional::NewGraph(cnn::ComputationGraph & cg) {
    for(auto & enc : encoders_)
        enc->NewGraph(cg);
    i_ehid_h_W_ = parameter(cg, p_ehid_h_W_);
    i_ehid_state_W_ = parameter(cg, p_ehid_state_W_);
    i_e_ehid_W_ = parameter(cg, p_e_ehid_W_);
    curr_graph_ = &cg;

}

ExternAttentional* ExternAttentional::Read(std::istream & in, cnn::Model & model) {
    int num_encoders, hidden_size, state_size;
    string version_id, line;
    if(!getline(in, line))
        THROW_ERROR("Premature end of model file when expecting ExternAttentional");
    istringstream iss(line);
    iss >> version_id >> num_encoders >> hidden_size >> state_size;
    if(version_id != "extatt_001")
        THROW_ERROR("Expecting a ExternAttentional of version extatt_001, but got something different:" << endl << line);
    vector<LinearEncoderPtr> encoders;
    while(num_encoders-- > 0)
        encoders.push_back(LinearEncoderPtr(LinearEncoder::Read(in, model)));
    return new ExternAttentional(encoders, hidden_size, state_size, model);
}
void ExternAttentional::Write(std::ostream & out) {
    out << "extatt_001 " << encoders_.size() << " " << hidden_size_ << " " << state_size_ << endl;
    for(auto & enc : encoders_) enc->Write(out);
}


void ExternAttentional::InitializeSentence(
            const Sentence & sent_src, bool train, cnn::ComputationGraph & cg) {

    sent_len_ = sent_src.size();

    // First get the states in a digestable format
    vector<vector<cnn::expr::Expression> > hs_sep;
    for(auto & enc : encoders_) {
        enc->BuildSentGraph(sent_src, train, cg);
        hs_sep.push_back(enc->GetWordStates());
    }
    // Concatenate them if necessary
    vector<cnn::expr::Expression> hs_comb;
    if(encoders_.size() == 1) {
        hs_comb = hs_sep[0];
    } else {
        for(int i : boost::irange(0, sent_len_)) {
            vector<cnn::expr::Expression> vars;
            for(int j : boost::irange(0, (int)encoders_.size()))
                vars.push_back(hs_sep[j][i]);
            hs_comb.push_back(concatenate(vars));
        }
    }
    if(hs_comb.size() >= 512) {
      cg.PrintGraphviz();
      THROW_ERROR("Oversized sentence combination (size="<<hs_comb.size()<<"): " << sent_src);
    }
    i_h_ = concatenate_cols(hs_comb);

    // TODO: Currently not using any bias
    // i_ehid_h_W_ is {hidden_size, context_size}, i_h_ is {context_size, sent_len}
    i_ehid_hpart_ = i_ehid_h_W_*i_h_;

    // Create an identity with shape
    sent_values_.resize(sent_len_, 1.0);
    i_sent_len_ = input(cg, {1, (unsigned int)sent_len_}, &sent_values_);

}


// Create a variable encoding the context
cnn::expr::Expression ExternAttentional::CreateContext(
        // const Sentence & sent, int loc,
        const std::vector<cnn::expr::Expression> & state_in,
        bool train,
        cnn::ComputationGraph & cg,
        std::vector<cnn::expr::Expression> & align_out) const {
    if(&cg != curr_graph_)
        THROW_ERROR("Initialized computation graph and passed comptuation graph don't match."); 
    cnn::expr::Expression i_ehid;
    if(state_in.size()) {
        // i_ehid_state_W_ is {hidden_size, state_size}, state_in is {state_size, 1}
        cnn::expr::Expression i_ehid_spart = i_ehid_state_W_ * *state_in.rbegin();
        i_ehid = affine_transform({i_ehid_hpart_, i_ehid_spart, i_sent_len_});
    } else {
        i_ehid = i_ehid_hpart_;
    }
    // Run through nonlinearity
    cnn::expr::Expression i_ehid_out = tanh({i_ehid});
    // i_e_ehid_W_ is {1, hidden_size}, i_ehid_out is {hidden_size, sent_len}
    cnn::expr::Expression i_e = i_e_ehid_W_ * i_ehid_out;
    cnn::expr::Expression i_e_trans = transpose(i_e);
    cnn::expr::Expression i_alpha = softmax({i_e_trans});
    align_out.push_back(i_alpha);
    // Print alignments
    if(GlobalVars::verbose >= 2) {
        vector<cnn::real> softmax = as_vector(cg.incremental_forward());
        cerr << "Alignments: " << softmax << endl;
    }
    // i_h_ is {input_size, sent_len}, i_alpha is {sent_len, 1}
    return i_h_ * i_alpha; 
}

EncoderAttentional::EncoderAttentional(
                   const ExternAttentionalPtr & extern_calc,
                   const NeuralLMPtr & decoder,
                   cnn::Model & model)
    : extern_calc_(extern_calc), decoder_(decoder), curr_graph_(NULL) { }


void EncoderAttentional::NewGraph(cnn::ComputationGraph & cg) {
    extern_calc_->NewGraph(cg);
    decoder_->NewGraph(cg);
    curr_graph_ = &cg;
}

cnn::expr::Expression EncoderAttentional::BuildSentGraph(
                    int sent_id,
                    const Sentence & sent_src, const Sentence & sent_trg,
                    bool train,
                    cnn::ComputationGraph & cg, LLStats & ll) {
    if(&cg != curr_graph_)
        THROW_ERROR("Initialized computation graph and passed comptuation graph don't match."); 
    extern_calc_->InitializeSentence(sent_src, train, cg);
    vector<cnn::expr::Expression> decoder_in;
    return decoder_->BuildSentGraph(sent_id, sent_trg, extern_calc_.get(), decoder_in, train, cg, ll);
}


EncoderAttentional* EncoderAttentional::Read(const DictPtr & vocab_src, const DictPtr & vocab_trg, std::istream & in, cnn::Model & model) {
    string version_id, line;
    if(!getline(in, line))
        THROW_ERROR("Premature end of model file when expecting EncoderAttentional");
    istringstream iss(line);
    iss >> version_id;
    if(version_id != "encatt_001")
        THROW_ERROR("Expecting a EncoderAttentional of version encatt_001, but got something different:" << endl << line);
    ExternAttentionalPtr extern_calc(ExternAttentional::Read(in, model));
    NeuralLMPtr decoder(NeuralLM::Read(vocab_trg, in, model));
    return new EncoderAttentional(extern_calc, decoder, model);
}
void EncoderAttentional::Write(std::ostream & out) {
    out << "encatt_001" << endl;
    extern_calc_->Write(out);
    decoder_->Write(out);
}
