#pragma once

#include <lamtram/encoder-classifier.h>
#include <dynet/tensor.h>
#include <dynet/dynet.h>
#include <vector>

namespace lamtram {

class EnsembleClassifier {

public:
    EnsembleClassifier(const std::vector<EncoderClassifierPtr> & encdecs);
    ~EnsembleClassifier() {}

    void CalcEval(const Sentence & sent_src, int trg, LLStats & ll);
    int Predict(const Sentence & sent_src);

    std::string GetEnsembleOperation() const { return ensemble_operation_; }
    void SetEnsembleOperation(const std::string & ensemble_operation) { ensemble_operation_ = ensemble_operation; }

    int MaxElement(const std::vector<dynet::real> & vals) const;

protected:
    std::vector<EncoderClassifierPtr> encclss_;
    std::string ensemble_operation_;

};

}
