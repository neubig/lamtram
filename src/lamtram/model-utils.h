#pragma once

#include <lamtram/dict-utils.h>
#include <dynet/dynet.h>
#include <iostream>
#include <memory>

namespace dynet {
class Model;
}

namespace lamtram {

class ModelUtils {
public:
    static void WriteModelText(std::ostream & out, const dynet::Model & mod);
    static void ReadModelText(std::istream & in, dynet::Model & mod);

    // Load a model from a stream
    // Will return a pointer to the model, and reset the passed shared pointers
    // with dynet::Model, and input, output vocabularies (if necessary)
    template <class ModelType>
    static ModelType* LoadBilingualModel(std::istream & in,
                                std::shared_ptr<dynet::Model> & mod,
                                DictPtr & vocab_src, DictPtr & vocab_trg);

    // Load a model from a text file
    template <class ModelType>
    static ModelType* LoadBilingualModel(const std::string & infile,
                                std::shared_ptr<dynet::Model> & mod,
                                DictPtr & vocab_src, DictPtr & vocab_trg);

    // Load a model from a stream
    // Will return a pointer to the model, and reset the passed shared pointers
    // with dynet::Model, and input, output vocabularies (if necessary)
    template <class ModelType>
    static ModelType* LoadMonolingualModel(std::istream & in,
                                std::shared_ptr<dynet::Model> & mod,
                                DictPtr & vocab_trg);

    // Load a model from a text file
    template <class ModelType>
    static ModelType* LoadMonolingualModel(const std::string & infile,
                                std::shared_ptr<dynet::Model> & mod,
                                DictPtr & vocab_trg);

};

}
