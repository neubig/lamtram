#include <lamtram/neural-lm.h>
#include <lamtram/macros.h>
#include <lamtram/builder-factory.h>
#include <lamtram/extern-calculator.h>
#include <lamtram/softmax-factory.h>
#include <cnn/dict.h>
#include <cnn/model.h>
#include <cnn/nodes.h>
#include <cnn/rnn.h>
#include <boost/range/irange.hpp>
#include <ctime>
#include <fstream>

using namespace std;
using namespace lamtram;

NeuralLM::NeuralLM(const DictPtr & vocab, int ngram_context, int extern_context, int wordrep_size,
                   const string & hidden_spec, int unk_id, const std::string & softmax_sig,
                   cnn::Model & model) :
            vocab_(vocab), ngram_context_(ngram_context),
            extern_context_(extern_context), wordrep_size_(wordrep_size),
            unk_id_(unk_id), hidden_spec_(hidden_spec), curr_graph_(NULL) {
    // Hidden layers
    builder_ = BuilderFactory::CreateBuilder(hidden_spec_,
                                             ngram_context*wordrep_size + extern_context,
                                             model);
    // Word representations
    p_wr_W_ = model.add_lookup_parameters(vocab->size(), {(unsigned int)wordrep_size}); 

    // Create the softmax
    softmax_ = SoftmaxFactory::CreateSoftmax(softmax_sig, hidden_spec_.nodes, vocab, model);
}

cnn::expr::Expression NeuralLM::BuildSentGraph(const Sentence & sent,
                                            const ExternCalculator * extern_calc,
                                            const std::vector<cnn::expr::Expression> & layer_in,
                                            bool train,
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
            extern_in = extern_calc->CreateContext(builder_->final_h(), train, cg, aligns);
            i_wrs_t.push_back(extern_in);
        }
        cnn::expr::Expression i_wr_t = concatenate(i_wrs_t);
        // Run the hidden unit
        cnn::expr::Expression i_h_t = builder_->add_input(i_wr_t);
        // Run the softmax and calculate the error
        cnn::expr::Expression i_err = softmax_->CalcLoss(i_h_t, sent[t], train);
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
    softmax_->NewGraph(cg);
    curr_graph_ = &cg;
}

inline unsigned CreateWord(const Sentence & sent, int t) {
  return t < (int)sent.size() ? sent[t] : 0;
}
inline vector<unsigned> CreateWord(const vector<Sentence> & sent, int t) {
  vector<unsigned> ret(sent.size());
  for(size_t i = 0; i < sent.size(); i++)
    ret[i] = CreateWord(sent[i], t);
  return ret;
}

// Move forward one step using the language model and return the probabilities
template <class Sent>
cnn::expr::Expression NeuralLM::Forward(const Sent & sent, int t, 
                                     const ExternCalculator * extern_calc,
                                     bool log_prob,
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
        i_wrs_t.push_back(lookup(cg, p_wr_W_, CreateWord(sent, hist)));
    if(extern_context_ > 0) {
        cnn::expr::Expression extern_in = extern_calc->CreateContext(layer_in, false, cg, align_out);
        i_wrs_t.push_back(extern_in);
    }
    cnn::expr::Expression i_wr_t = concatenate(i_wrs_t);
    // Run the hidden unit
    cnn::expr::Expression i_h_t = builder_->add_input(i_wr_t);
    // Run the softmax and calculate the error
    cnn::expr::Expression i_sm_t = (log_prob ?
                                    softmax_->CalcLogProbability(i_h_t) :
                                    softmax_->CalcProbability(i_h_t));
    // Update the state
    layer_out = builder_->final_s();
    return i_sm_t;
}

// Instantiate
template
cnn::expr::Expression NeuralLM::Forward<Sentence>(
                                     const Sentence & sent, int t, 
                                     const ExternCalculator * extern_calc,
                                     bool log_prob,
                                     const std::vector<cnn::expr::Expression> & layer_in,
                                     std::vector<cnn::expr::Expression> & layer_out,
                                     cnn::ComputationGraph & cg,
                                     std::vector<cnn::expr::Expression> & align_out);
template
cnn::expr::Expression NeuralLM::Forward<vector<Sentence> >(
                                     const vector<Sentence> & sent, int t, 
                                     const ExternCalculator * extern_calc,
                                     bool log_prob,
                                     const std::vector<cnn::expr::Expression> & layer_in,
                                     std::vector<cnn::expr::Expression> & layer_out,
                                     cnn::ComputationGraph & cg,
                                     std::vector<cnn::expr::Expression> & align_out);

NeuralLM* NeuralLM::Read(const DictPtr & vocab, std::istream & in, cnn::Model & model) {
    int vocab_size, ngram_context, extern_context = 0, wordrep_size, unk_id;
    string version_id, hidden_spec, line, softmax_sig;
    if(!getline(in, line))
        THROW_ERROR("Premature end of model file when expecting Neural LM");
    istringstream iss(line);
    iss >> version_id;
    if(version_id == "nlm_002") {
        iss >> vocab_size >> ngram_context >> wordrep_size >> hidden_spec >> unk_id;
    } else if(version_id == "nlm_003") {
        iss >> vocab_size >> ngram_context >> extern_context >> wordrep_size >> hidden_spec >> unk_id;
    } else if(version_id == "nlm_004") {
        iss >> vocab_size >> ngram_context >> extern_context >> wordrep_size >> hidden_spec >> unk_id >> softmax_sig;
    } else {
        THROW_ERROR("Expecting a Neural LM of version nlm_{002,003,004}, but got something different:" << endl << line);
    }
    assert(vocab->size() == vocab_size);
    return new NeuralLM(vocab, ngram_context, extern_context, wordrep_size, hidden_spec, unk_id, softmax_sig, model);
}
void NeuralLM::Write(std::ostream & out) {
    out << "nlm_004 " << vocab_->size() << " " << ngram_context_ << " " << extern_context_ << " " << wordrep_size_ << " " << hidden_spec_ << " " << unk_id_ << " " << softmax_->GetSig() << endl;
}

int NeuralLM::GetVocabSize() const { return vocab_->size(); }

