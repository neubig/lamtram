#pragma once

#include <dynet/tensor.h>
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

    dynet::real CalcSentLoss() {
        return loss_/sents_;
    }

    int sents_; // Number of sents
    dynet::real loss_;  // Log losselihood

};

}
