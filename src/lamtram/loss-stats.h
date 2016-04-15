#pragma once

#include <cnn/tensor.h>
#include <cmath>

namespace lamtram {

class LossStats {

public:
    LossStats(const LossStats & rhs) : sents_(rhs.sents_), loss_(rhs.loss_) { }
    LossStats() : sents_(0), loss_(0.0) { }

    LossStats & operator+=(const LossStats & rhs) {
        sents_ += rhs.sents_;
        loss_ += rhs.loss_;
        return *this;
    }

    cnn::real CalcSentLoss() {
        return loss_/sents_;
    }

    int sents_; // Number of sents
    cnn::real loss_;  // Log losselihood

};

}
