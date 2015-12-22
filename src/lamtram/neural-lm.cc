#include <lamtram/neural-lm.h>
#include <lamtram/vocabulary.h>
#include <lamtram/macros.h>
#include <lamtram/builder-factory.h>
#include <lamtram/extern-calculator.h>
#include <cnn/model.h>
#include <cnn/nodes.h>
#include <cnn/rnn.h>
#include <boost/range/irange.hpp>
#include <ctime>
#include <fstream>

using namespace std;
using namespace lamtram;

NeuralLM::NeuralLM(int vocab_size, int ngram_context, int extern_context, int wordrep_size,
                   const string & hidden_spec, int unk_id,
                   cnn::Model & model) :
            vocab_size_(vocab_size), ngram_context_(ngram_context),
            extern_context_(extern_context), wordrep_size_(wordrep_size),
            unk_id_(unk_id), hidden_spec_(hidden_spec), curr_graph_(NULL) {
    // Hidden layers
    builder_ = BuilderFactory::CreateBuilder(hidden_spec_,
                                             ngram_context*wordrep_size + extern_context,
                                             model);
    // Word representations
    p_wr_W_ = model.add_lookup_parameters(vocab_size, {(unsigned int)wordrep_size}); 
    // Softmax and its corresponding bias
    p_sm_W_ = model.add_parameters({(unsigned int)vocab_size, (unsigned int)hidden_spec_.nodes});
    p_sm_b_ = model.add_parameters({(unsigned int)vocab_size});
}

cnn::expr::Expression NeuralLM::BuildSentGraph(const Sentence & sent,
                                            const ExternCalculator * extern_calc,
                                            const std::vector<cnn::expr::Expression> & layer_in,
                                            cnn::ComputationGraph & cg, LLStats & ll) {
    if(&cg != curr_graph_)
        THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
    int slen = sent.size() - 1;
    builder_->start_new_sequence(layer_in);
    // First get all the word representations
    vector<cnn::expr::Expression> i_wr;
    for(auto t : boost::irange(0, slen))
        i_wr.push_back(lookup(cg, p_wr_W_, sent[t]));
    // Next, do the computation
    vector<cnn::expr::Expression> errs, aligns;
    for(auto t : boost::irange(ngram_context_, slen+1)) {
        // Concatenate wordrep and external context into a vector for the hidden unit
        vector<cnn::expr::Expression> i_wrs_t;
        for(auto hist : boost::irange(t - ngram_context_, t))
            i_wrs_t.push_back(i_wr[hist]);
        if(extern_context_ > 0) {
            assert(extern_calc != NULL);
            cnn::expr::Expression extern_in;
            extern_in = extern_calc->CreateContext(sent, t, builder_->final_h(), cg, aligns);
            i_wrs_t.push_back(extern_in);
        }
        cnn::expr::Expression i_wr_t = concatenate(i_wrs_t);
        // Run the hidden unit
        cnn::expr::Expression i_h_t = builder_->add_input(i_wr_t);
        // Run the softmax and calculate the error
        cnn::expr::Expression i_sm_t = affine_transform({i_sm_b_, i_sm_W_, i_h_t});
        cnn::expr::Expression i_err = pickneglogsoftmax({i_sm_t}, sent[t]);
        errs.push_back(i_err);
        // If this word is unknown, then add to the unknown count
        if(sent[t] == unk_id_)
            ll.unk_++;
        ll.words_++;
    }
    cnn::expr::Expression i_nerr = sum(errs);
    return i_nerr;
}

void NeuralLM::NewGraph(cnn::ComputationGraph & cg) {
    builder_->new_graph(cg);
    builder_->start_new_sequence();
    i_sm_b_ = parameter(cg, p_sm_b_);
    i_sm_W_ = parameter(cg, p_sm_W_);
    curr_graph_ = &cg;
}

// Move forward one step using the language model and return the probabilities
template <class SoftmaxOp>
cnn::expr::Expression NeuralLM::Forward(const Sentence & sent, int t, 
                                     const ExternCalculator * extern_calc,
                                     const std::vector<cnn::expr::Expression> & layer_in,
                                     std::vector<cnn::expr::Expression> & layer_out,
                                     cnn::ComputationGraph & cg,
                                     std::vector<cnn::expr::Expression> & align_out) {
    if(&cg != curr_graph_)
        THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
    // Start a new sequence if necessary
    if(layer_in.size())
        builder_->start_new_sequence(layer_in);
    // Concatenate wordrep and external context into a vector for the hidden unit
    vector<cnn::expr::Expression> i_wrs_t;
    for(auto hist : boost::irange(t - ngram_context_, t))
        i_wrs_t.push_back(lookup(cg, p_wr_W_, sent[hist]));
    if(extern_context_ > 0) {
        cnn::expr::Expression extern_in = extern_calc->CreateContext(sent, t, layer_in, cg, align_out);
        i_wrs_t.push_back(extern_in);
    }
    cnn::expr::Expression i_wr_t = concatenate(i_wrs_t);
    // Run the hidden unit
    cnn::expr::Expression i_h_t = builder_->add_input(i_wr_t);
    // Run the softmax and calculate the error
    cnn::expr::Expression i_smin_t = affine_transform({i_sm_b_, i_sm_W_, i_h_t});
    cnn::expr::Expression i_sm_t = cnn::expr::Expression(i_smin_t.pg, i_smin_t.pg->add_function<SoftmaxOp>({i_smin_t.i}));
    // Update the state
    layer_out = builder_->final_s();
    return i_sm_t;
}

// Instantiate
template
cnn::expr::Expression NeuralLM::Forward<cnn::Softmax>(
                                     const Sentence & sent, int t, 
                                     const ExternCalculator * extern_calc,
                                     const std::vector<cnn::expr::Expression> & layer_in,
                                     std::vector<cnn::expr::Expression> & layer_out,
                                     cnn::ComputationGraph & cg,
                                     std::vector<cnn::expr::Expression> & align_out);
template
cnn::expr::Expression NeuralLM::Forward<cnn::LogSoftmax>(
                                     const Sentence & sent, int t, 
                                     const ExternCalculator * extern_calc,
                                     const std::vector<cnn::expr::Expression> & layer_in,
                                     std::vector<cnn::expr::Expression> & layer_out,
                                     cnn::ComputationGraph & cg,
                                     std::vector<cnn::expr::Expression> & align_out);

NeuralLM* NeuralLM::Read(std::istream & in, cnn::Model & model) {
    int vocab_size, ngram_context, extern_context = 0, wordrep_size, unk_id;
    string version_id, hidden_spec, line;
    if(!getline(in, line))
        THROW_ERROR("Premature end of model file when expecting Neural LM");
    istringstream iss(line);
    iss >> version_id;
    if(version_id == "nlm_002") {
        iss >> vocab_size >> ngram_context >> wordrep_size >> hidden_spec >> unk_id;
    } else if(version_id == "nlm_003") {
        iss >> vocab_size >> ngram_context >> extern_context >> wordrep_size >> hidden_spec >> unk_id;
    } else {
        THROW_ERROR("Expecting a Neural LM of version nlm_{002,003}, but got something different:" << endl << line);
    }
    return new NeuralLM(vocab_size, ngram_context, extern_context, wordrep_size, hidden_spec, unk_id, model);
}
void NeuralLM::Write(std::ostream & out) {
    out << "nlm_003 " << vocab_size_ << " " << ngram_context_ << " " << extern_context_ << " " << wordrep_size_ << " " << hidden_spec_ << " " << unk_id_ << endl;
}
