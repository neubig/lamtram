#pragma once

#include <dynet/tensor.h>
#include <cmath>
#include <sstream>

namespace lamtram {

class LLStats {

public:
    LLStats(const LLStats & rhs) : vocab_(rhs.vocab_), words_(rhs.words_), unk_(rhs.unk_), loss_(rhs.loss_), is_likelihood_(rhs.is_likelihood_) { }
    LLStats(int vocab) : vocab_(vocab), words_(0), unk_(0), correct_(0), loss_(0.0), is_likelihood_(true) { }

    LLStats & operator+=(const LLStats & rhs) {
        words_ += rhs.words_;
        unk_ += rhs.unk_;
        loss_ += rhs.loss_;
        correct_ += rhs.correct_;
        return *this;
    }

    dynet::real CalcAcc() {
        return correct_/(dynet::real)words_;
    }

    dynet::real CalcAvgLoss() {
        return loss_/words_;
    }
    dynet::real CalcUnkLoss() {
        return loss_+unk_*log(vocab_);
    }

    dynet::real CalcPPL() {
        return pow(2, (CalcUnkLoss()/words_/log(2)));
    }
    dynet::real CalcPPLNoUnk() {
        return pow(2, (loss_/words_/log(2)));
    }

    std::string PrintStats() {
      std::ostringstream oss;
      if(is_likelihood_) {
        oss << "ppl=" << CalcPPL() << ", unk=" << unk_;
      } else {
        oss << "loss=" << CalcAvgLoss() << ", unk=" << unk_;
      }
      return oss.str();
    }

    int vocab_; // Vocabulary size
    int words_; // Number of words
    int unk_;   // Number of unknown words
    int correct_; // Number of correct predictions
    dynet::real loss_;  // Loss
    bool is_likelihood_; // Whether this represents a likelihood, or a generic loss

};

}
