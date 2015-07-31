#include <lamtram/ensemble-classifier.h>
#include <lamtram/macros.h>
#include <cnn/nodes.h>
#include <boost/range/irange.hpp>
#include <cfloat>

using namespace lamtram;
using namespace std;
using namespace cnn::expr;


EnsembleClassifier::EnsembleClassifier(const vector<EncoderClassifierPtr> & encclss)
            : encclss_(encclss), ensemble_operation_("sum") {
    if(encclss.size() == 0)
        THROW_ERROR("Cannot decode with no models!");
}

void EnsembleClassifier::CalcEval(const Sentence & sent_src, int trg, LLStats & ll) {
    // First initialize states and do encoding as necessary
    cnn::ComputationGraph cg;
    vector<Expression> i_sms;
    for(auto & tm : encclss_) {
        tm->NewGraph(cg);
        if(ensemble_operation_ == "sum") {
            i_sms.push_back(tm->Forward<cnn::Softmax>(sent_src, cg));
        } else {
            i_sms.push_back(tm->Forward<cnn::LogSoftmax>(sent_src, cg));
        }
    }
    Expression i_average = average(i_sms);
    int label = MaxElement(cnn::as_vector(cg.forward()));
    // Ensemble the probabilities and calculate the likelihood
    Expression i_logprob;
    if(ensemble_operation_ == "sum") {
        i_logprob = log(pick({i_average}, trg));
    } else if(ensemble_operation_ == "logsum") {
        i_logprob = pick({log_softmax(i_average)}, trg);
    } else {
        THROW_ERROR("Bad ensembling operation: " << ensemble_operation_ << endl);
    }
    ll.lik_ += as_scalar(cg.incremental_forward());
    // Check if it's correct
    if(label == trg) ll.correct_++;
    ll.words_++;
}                                        

int EnsembleClassifier::Predict(const Sentence & sent_src) {
    // First initialize states and do encoding as necessary
    cnn::ComputationGraph cg;
    vector<Expression> i_sms;
    for(auto & tm : encclss_) {
        tm->NewGraph(cg);
        if(ensemble_operation_ == "sum") {
            i_sms.push_back(tm->Forward<cnn::Softmax>(sent_src, cg));
        } else {
            i_sms.push_back(tm->Forward<cnn::LogSoftmax>(sent_src, cg));
        }
    }
    sum(i_sms);
    return MaxElement(cnn::as_vector(cg.forward()));
}


int EnsembleClassifier::MaxElement(const std::vector<cnn::real> & vals) const {
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
