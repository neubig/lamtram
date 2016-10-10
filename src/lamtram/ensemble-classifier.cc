#include <lamtram/ensemble-classifier.h>
#include <lamtram/macros.h>
#include <dynet/nodes.h>
#include <boost/range/irange.hpp>
#include <cfloat>

using namespace lamtram;
using namespace std;
using namespace dynet::expr;


EnsembleClassifier::EnsembleClassifier(const vector<EncoderClassifierPtr> & encclss)
            : encclss_(encclss), ensemble_operation_("sum") {
    if(encclss.size() == 0)
        THROW_ERROR("Cannot decode with no models!");
}

void EnsembleClassifier::CalcEval(const Sentence & sent_src, int trg, LLStats & ll) {
    // First initialize states and do encoding as necessary
    dynet::ComputationGraph cg;
    vector<Expression> i_sms;
    for(auto & tm : encclss_) {
        tm->NewGraph(cg);
        // TODO: add identity
        if(ensemble_operation_ == "sum") {
            i_sms.push_back(tm->Forward<dynet::Softmax>(sent_src, false, cg));
        } else {
            i_sms.push_back(tm->Forward<dynet::LogSoftmax>(sent_src, false, cg));
        }
    }
    Expression i_average = average(i_sms);
    int label = MaxElement(dynet::as_vector(cg.incremental_forward(i_average)));
    // Ensemble the probabilities and calculate the likelihood
    Expression i_logprob;
    if(ensemble_operation_ == "sum") {
        i_logprob = log(pick({i_average}, trg));
    } else if(ensemble_operation_ == "logsum") {
        i_logprob = pick({log_softmax(i_average)}, trg);
    } else {
        THROW_ERROR("Bad ensembling operation: " << ensemble_operation_ << endl);
    }
    ll.loss_ -= as_scalar(cg.incremental_forward(i_logprob));
    // Check if it's correct
    if(label == trg) ll.correct_++;
    ll.words_++;
}                                        

int EnsembleClassifier::Predict(const Sentence & sent_src) {
    // First initialize states and do encoding as necessary
    dynet::ComputationGraph cg;
    vector<Expression> i_sms;
    for(auto & tm : encclss_) {
        tm->NewGraph(cg);
        if(ensemble_operation_ == "sum") {
            i_sms.push_back(tm->Forward<dynet::Softmax>(sent_src, false, cg));
        } else {
            i_sms.push_back(tm->Forward<dynet::LogSoftmax>(sent_src, false, cg));
        }
    }
    dynet::expr::Expression prob_exp = sum(i_sms);
    return MaxElement(dynet::as_vector(cg.incremental_forward(prob_exp)));
}


int EnsembleClassifier::MaxElement(const std::vector<dynet::real> & vals) const {
    if(!vals.size()) THROW_ERROR("Can't get max element of empty vector");
    int best_id = 0;
    float best_val = vals[0];
    for(int i : boost::irange(1, (int)vals.size())) {
        if(best_val < vals[i]) {
            best_id = i;
            best_val = vals[i];
        }
    }
    return best_id;
}
