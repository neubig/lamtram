#pragma once

#include <lamtram/vocabulary.h>
#include <cnn/cnn.h>
#include <iostream>
#include <memory>

namespace cnn {
class Model;
}

namespace lamtram {

class ModelUtils {
public:
    static void WriteModelText(std::ostream & out, const cnn::Model & mod);
    static void ReadModelText(std::istream & in, cnn::Model & mod);

    // Load a model from a text file
    // Will return a pointer to the model, and reset the passed shared pointers
    // with cnn::Model, and input, output vocabularies (if necessary)
    template <class ModelType>
    static ModelType* LoadModel(const std::string & infile,
                                std::shared_ptr<cnn::Model> & mod,
                                VocabularyPtr & vocab_src, VocabularyPtr & vocab_trg);

};

}
