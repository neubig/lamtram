#include <lamtram/linear-encoder.h>
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

LinearEncoder::LinearEncoder(int vocab_size, int wordrep_size,
                   const string & hidden_spec, int unk_id,
                   cnn::Model & model) :
            vocab_size_(vocab_size), wordrep_size_(wordrep_size), unk_id_(unk_id), hidden_spec_(hidden_spec), reverse_(false) {
    // Hidden layers
    builder_ = BuilderFactory::CreateBuilder(hidden_spec_, wordrep_size, model);
    // Word representations
    p_wr_W_ = model.add_lookup_parameters(vocab_size, {wordrep_size}); 
}

cnn::expr::Expression LinearEncoder::BuildSentGraph(const Sentence & sent, cnn::ComputationGraph & cg) {
    if(&cg != curr_graph_)
        THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
    word_states_.resize(sent.size());
    builder_->start_new_sequence();
    // First get all the word representations
    cnn::expr::Expression i_wr_t, i_h_t;
    if(!reverse_) {
        for(int t = 0; t < (int)sent.size(); t++) {
            i_wr_t = lookup(cg, p_wr_W_, sent[t]);
            i_h_t = builder_->add_input(i_wr_t);
            word_states_[t] = i_h_t;
        }
    } else {
        for(int t = sent.size()-1; t >= 0; t--) {
            i_wr_t = lookup(cg, p_wr_W_, sent[t]);
            i_h_t = builder_->add_input(i_wr_t);
            word_states_[t] = i_h_t;
        }
    }
    return i_h_t;
}


void LinearEncoder::NewGraph(cnn::ComputationGraph & cg) {
    builder_->new_graph(cg);
    curr_graph_ = &cg;
}

void LinearEncoder::CopyParameters(const LinearEncoder & enc) {
    assert(vocab_size_ == enc.vocab_size_);
    assert(wordrep_size_ == enc.wordrep_size_);
    assert(reverse_ == enc.reverse_);
    p_wr_W_->copy(*enc.p_wr_W_);
    builder_->copy(*enc.builder_);
}

LinearEncoder* LinearEncoder::Read(std::istream & in, cnn::Model & model) {
    int vocab_size, wordrep_size, unk_id;
    string version_id, hidden_spec, line, reverse;
    if(!getline(in, line))
        THROW_ERROR("Premature end of model file when expecting Neural LM");
    istringstream iss(line);
    iss >> version_id >> vocab_size >> wordrep_size >> hidden_spec >> unk_id >> reverse;
    if(version_id != "linenc_001")
        THROW_ERROR("Expecting a Neural LM of version linenc_001, but got something different:" << endl << line);
    LinearEncoder * ret = new LinearEncoder(vocab_size, wordrep_size, hidden_spec, unk_id, model);
    if(reverse == "rev") ret->SetReverse(true);
    return ret;
}
void LinearEncoder::Write(std::ostream & out) {
    out << "linenc_001 " << vocab_size_ << " " << wordrep_size_ << " " << hidden_spec_ << " " << unk_id_ << " " << (reverse_?"rev":"for") << endl;
}

vector<cnn::expr::Expression> LinearEncoder::GetFinalHiddenLayers() const {
    return builder_->final_h();
}
