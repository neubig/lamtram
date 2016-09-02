
#include <lamtram/model-utils.h>
#include <lamtram/macros.h>
#include <lamtram/encoder-decoder.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/encoder-classifier.h>
#include <lamtram/neural-lm.h>
#include <cnn/model.h>
#include <cnn/dict.h>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <fstream>

using namespace std;
using namespace lamtram;

void ModelUtils::WriteModelText(ostream & out, const cnn::Model & mod) {
    boost::archive::text_oarchive oa(out);
    oa << mod;
}
void ModelUtils::ReadModelText(istream & in, cnn::Model & mod) {
    boost::archive::text_iarchive ia(in);
    ia >> mod;
}


// Load a model from a stream
// Will return a pointer to the model, and reset the passed shared pointers
// with cnn::Model, and input, output vocabularies (if necessary)
template <class ModelType>
ModelType* ModelUtils::LoadBilingualModel(std::istream & model_in,
                                          std::shared_ptr<cnn::Model> & mod,
                                          DictPtr & vocab_src, DictPtr & vocab_trg) {
    vocab_src.reset(ReadDict(model_in));
    vocab_trg.reset(ReadDict(model_in));
    mod.reset(new cnn::Model);
    ModelType* ret = ModelType::Read(vocab_src, vocab_trg, model_in, *mod);
    ModelUtils::ReadModelText(model_in, *mod);
    return ret;
}

// Load a model from a text file
// Will return a pointer to the model, and reset the passed shared pointers
// with cnn::Model, and input, output vocabularies (if necessary)
template <class ModelType>
ModelType* ModelUtils::LoadBilingualModel(const std::string & file,
                                          std::shared_ptr<cnn::Model> & mod,
                                          DictPtr & vocab_src, DictPtr & vocab_trg) {
    ifstream model_in(file);
    if(!model_in) THROW_ERROR("Could not open model file " << file);
    return ModelUtils::LoadBilingualModel<ModelType>(model_in, mod, vocab_src, vocab_trg);
}

// Load a model from a npz file
// Will return a pointer to the model, and reset the passed shared pointers
// with cnn::Model, and input, output vocabularies (if necessary)
template <class ModelType>
ModelType* ModelUtils::ConvertBilingualModel(const std::string & infile,
                                          std::shared_ptr<cnn::Model> & mod,
                                          DictPtr & vocab_src, DictPtr & vocab_trg) {
    
    //get information from json file
    std::ifstream jsonFile(infile+".json");

    boost::property_tree::ptree pt;
    read_json(jsonFile, pt);

    //load dictionaries
    int source_voc_size= pt.get_child("n_words_src").get_value<int>();
    int target_voc_size= pt.get_child("n_words").get_value<int>();

    std::vector<std::string> dicts;
    for (boost::property_tree::ptree::value_type &d : pt.get_child("dictionaries"))
    {
        dicts.push_back(d.second.data());
    }

    vocab_src.reset(ConvertDict(dicts[0],source_voc_size));
    vocab_trg.reset(ConvertDict(dicts[1],target_voc_size));

    mod.reset(new cnn::Model);
    ModelType* ret = ModelType::Convert(vocab_src, vocab_trg, infile,pt, *mod);
    return ret;
}



// Load a model from a stream
// Will return a pointer to the model, and reset the passed shared pointers
// with cnn::Model, and input, output vocabularies (if necessary)
template <class ModelType>
ModelType* ModelUtils::LoadMonolingualModel(std::istream & model_in,
                                          std::shared_ptr<cnn::Model> & mod,
                                          DictPtr & vocab_trg) {
    vocab_trg.reset(ReadDict(model_in));
    mod.reset(new cnn::Model);
    ModelType* ret = ModelType::Read(vocab_trg, model_in, *mod);
    ModelUtils::ReadModelText(model_in, *mod);
    return ret;
}

// Load a model from a text file
// Will return a pointer to the model, and reset the passed shared pointers
// with cnn::Model, and input, output vocabularies (if necessary)
template <class ModelType>
ModelType* ModelUtils::LoadMonolingualModel(const std::string & file,
                                          std::shared_ptr<cnn::Model> & mod,
                                          DictPtr & vocab_trg) {
    ifstream model_in(file);
    if(!model_in) THROW_ERROR("Could not open model file " << file);
    return ModelUtils::LoadMonolingualModel<ModelType>(model_in, mod, vocab_trg);
}

// Instantiate LoadModel
template
EncoderDecoder* ModelUtils::LoadBilingualModel<EncoderDecoder>(std::istream & model_in,
                                                      std::shared_ptr<cnn::Model> & mod,
                                                      DictPtr & vocab_src, DictPtr & vocab_trg);
template
EncoderAttentional* ModelUtils::LoadBilingualModel<EncoderAttentional>(std::istream & model_in,
                                                              std::shared_ptr<cnn::Model> & mod,
                                                              DictPtr & vocab_src, DictPtr & vocab_trg);
template
EncoderClassifier* ModelUtils::LoadBilingualModel<EncoderClassifier>(std::istream & model_in,
                                                            std::shared_ptr<cnn::Model> & mod,
                                                            DictPtr & vocab_src, DictPtr & vocab_trg);
template
NeuralLM* ModelUtils::LoadMonolingualModel<NeuralLM>(std::istream & model_in,
                                          std::shared_ptr<cnn::Model> & mod,
                                          DictPtr & vocab_trg);
template
EncoderDecoder* ModelUtils::LoadBilingualModel<EncoderDecoder>(const std::string & infile,
                                                      std::shared_ptr<cnn::Model> & mod,
                                                      DictPtr & vocab_src, DictPtr & vocab_trg);
template
EncoderAttentional* ModelUtils::LoadBilingualModel<EncoderAttentional>(const std::string & infile,
                                                              std::shared_ptr<cnn::Model> & mod,
                                                              DictPtr & vocab_src, DictPtr & vocab_trg);
template
EncoderClassifier* ModelUtils::LoadBilingualModel<EncoderClassifier>(const std::string & infile,
                                                            std::shared_ptr<cnn::Model> & mod,
                                                            DictPtr & vocab_src, DictPtr & vocab_trg);
template
NeuralLM* ModelUtils::LoadMonolingualModel<NeuralLM>(const std::string & infile,
                                                     std::shared_ptr<cnn::Model> & mod,
                                                     DictPtr & vocab_trg);
template
EncoderAttentional* ModelUtils::ConvertBilingualModel<EncoderAttentional>(const std::string & infile,
                                                              std::shared_ptr<cnn::Model> & mod,
                                                              DictPtr & vocab_src, DictPtr & vocab_trg);
