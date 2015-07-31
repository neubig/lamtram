#pragma once

#include <cnn/tensor.h>
#include <cmath>

namespace lamtram {

class LLStats {

public:
    LLStats(const LLStats & rhs) : vocab_(rhs.vocab_), words_(rhs.words_), unk_(rhs.unk_), lik_(rhs.lik_) { }
    LLStats(int vocab) : vocab_(vocab), words_(0), unk_(0), correct_(0), lik_(0.0) { }

    LLStats & operator+=(const LLStats & rhs) {
        words_ += rhs.words_;
        unk_ += rhs.unk_;
        lik_ += rhs.lik_;
        correct_ += rhs.correct_;
        return *this;
    }

    cnn::real CalcAcc() {
        return correct_/(cnn::real)words_;
    }

    cnn::real CalcUnkLik() {
        return lik_-unk_*log(vocab_);
    }

    cnn::real CalcPPL() {
        return pow(2, (-CalcUnkLik()/words_/log(2)));
    }
    cnn::real CalcPPLNoUnk() {
        return pow(2, (-lik_/words_/log(2)));
    }

    int vocab_; // Vocabulary size
    int words_; // Number of words
    int unk_;   // Number of unknown words
    int correct_; // Number of correct predictions
    cnn::real lik_;  // Log likelihood

};

}
