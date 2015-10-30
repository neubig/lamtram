
#include <lamtram/model-utils.h>
#include <lamtram/macros.h>
#include <lamtram/encoder-decoder.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/encoder-classifier.h>
#include <lamtram/neural-lm.h>
#include <cnn/model.h>
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
ModelType* ModelUtils::LoadModel(std::istream & model_in,
                                 std::shared_ptr<cnn::Model> & mod,
                                 VocabularyPtr & vocab_src, VocabularyPtr & vocab_trg) {
    if(ModelType::HasSrcVocab()) {
        vocab_src.reset(Vocabulary::Read(model_in));
        vocab_src->SetFreeze(true);
        vocab_src->SetDefault("<unk>");
    }
    vocab_trg.reset(Vocabulary::Read(model_in));
    vocab_trg->SetFreeze(true);
    vocab_trg->SetDefault("<unk>");
    mod.reset(new cnn::Model);
    ModelType* ret = ModelType::Read(model_in, *mod);
    ModelUtils::ReadModelText(model_in, *mod);
    return ret;
}

// Load a model from a text file
// Will return a pointer to the model, and reset the passed shared pointers
// with cnn::Model, and input, output vocabularies (if necessary)
template <class ModelType>
ModelType* ModelUtils::LoadModel(const std::string & file,
                                 std::shared_ptr<cnn::Model> & mod,
                                 VocabularyPtr & vocab_src, VocabularyPtr & vocab_trg) {
    ifstream model_in(file);
    if(!model_in) THROW_ERROR("Could not open model file " << file);
    return ModelUtils::LoadModel<ModelType>(model_in, mod, vocab_src, vocab_trg);
}

// Instantiate LoadModel
template
EncoderDecoder* ModelUtils::LoadModel<EncoderDecoder>(std::istream & model_in,
                                                      std::shared_ptr<cnn::Model> & mod,
                                                      VocabularyPtr & vocab_src, VocabularyPtr & vocab_trg);
template
EncoderAttentional* ModelUtils::LoadModel<EncoderAttentional>(std::istream & model_in,
                                                              std::shared_ptr<cnn::Model> & mod,
                                                              VocabularyPtr & vocab_src, VocabularyPtr & vocab_trg);
template
EncoderClassifier* ModelUtils::LoadModel<EncoderClassifier>(std::istream & model_in,
                                                            std::shared_ptr<cnn::Model> & mod,
                                                            VocabularyPtr & vocab_src, VocabularyPtr & vocab_trg);
template
NeuralLM* ModelUtils::LoadModel<NeuralLM>(std::istream & model_in,
                                          std::shared_ptr<cnn::Model> & mod,
                                          VocabularyPtr & vocab_src, VocabularyPtr & vocab_trg);
template
EncoderDecoder* ModelUtils::LoadModel<EncoderDecoder>(const std::string & infile,
                                                      std::shared_ptr<cnn::Model> & mod,
                                                      VocabularyPtr & vocab_src, VocabularyPtr & vocab_trg);
template
EncoderAttentional* ModelUtils::LoadModel<EncoderAttentional>(const std::string & infile,
                                                              std::shared_ptr<cnn::Model> & mod,
                                                              VocabularyPtr & vocab_src, VocabularyPtr & vocab_trg);
template
EncoderClassifier* ModelUtils::LoadModel<EncoderClassifier>(const std::string & infile,
                                                            std::shared_ptr<cnn::Model> & mod,
                                                            VocabularyPtr & vocab_src, VocabularyPtr & vocab_trg);
template
NeuralLM* ModelUtils::LoadModel<NeuralLM>(const std::string & infile,
                                          std::shared_ptr<cnn::Model> & mod,
                                          VocabularyPtr & vocab_src, VocabularyPtr & vocab_trg);
