#include <lamtram/encoder-classifier.h>
#include <lamtram/macros.h>
#include <dynet/model.h>
#include <dynet/nodes.h>
#include <fstream>

using namespace std;
using namespace lamtram;
using namespace dynet::expr;

EncoderClassifier::EncoderClassifier(
                   const vector<LinearEncoderPtr> & encoders,
                   const ClassifierPtr & classifier,
                   dynet::Model & model) : encoders_(encoders), classifier_(classifier), curr_graph_(NULL) {
    // Encoder to classifier mapping parameters
    int enc2cls_in = 0;
    for(auto & enc : encoders)
        enc2cls_in += enc->GetNumLayers() * enc->GetNumNodes();
    int enc2cls_out = classifier_->GetInputSize();
    p_enc2cls_W_ = model.add_parameters({(unsigned int)enc2cls_out, (unsigned int)enc2cls_in});
    p_enc2cls_b_ = model.add_parameters({(unsigned int)enc2cls_out});
}


void EncoderClassifier::NewGraph(dynet::ComputationGraph & cg) {
    for(auto & enc : encoders_)
        enc->NewGraph(cg);
    classifier_->NewGraph(cg);
    i_enc2cls_b_ = parameter(cg, p_enc2cls_b_);
    i_enc2cls_W_ = parameter(cg, p_enc2cls_W_);
    curr_graph_ = &cg;
}

template <class SentData>
dynet::expr::Expression EncoderClassifier::GetEncodedState(
                        const SentData & sent_src, bool train, dynet::ComputationGraph & cg) const {
    // Perform encoding with each encoder
    vector<dynet::expr::Expression> inputs;
    for(auto & enc : encoders_) {
        enc->BuildSentGraph(sent_src, true, train, cg);
        for(auto & id : enc->GetFinalHiddenLayers())
            inputs.push_back(id);
    }
    // Perform transformation
    dynet::expr::Expression i_combined;
    assert(inputs.size() > 0);
    if(inputs.size() == 1) { i_combined = inputs[0]; }
    else                   { i_combined = concatenate(inputs); }
    return tanh(affine_transform({i_enc2cls_b_, i_enc2cls_W_, i_combined}));
}

dynet::expr::Expression EncoderClassifier::BuildSentGraph(const Sentence & sent_src,
                                                        const int & trg,
                                                        const int & cache,
                                                        const float * weight,
                                                        float samp_percent,
                                                        bool train,
                                                        dynet::ComputationGraph & cg,
                                                        LLStats & ll) {
    if(&cg != curr_graph_)
        THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
    // Perform encoding with each encoder
    dynet::expr::Expression classifier_in = GetEncodedState(sent_src, train, cg);
    ll.words_ += classifier_in.value().d.bd;
    return classifier_->BuildGraph(classifier_in, trg, train, cg);
}

dynet::expr::Expression EncoderClassifier::BuildSentGraph(const std::vector<Sentence> & sent_src,
                                                        const std::vector<int> & trg,
                                                        const std::vector<int> & cache,
                                                        const std::vector<float> * weights,
                                                        float samp_percent,
                                                        bool train,
                                                        dynet::ComputationGraph & cg,
                                                        LLStats & ll) {
    if(&cg != curr_graph_)
        THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
    // Perform encoding with each encoder
    dynet::expr::Expression classifier_in = GetEncodedState(sent_src, train, cg);
    ll.words_ += classifier_in.value().d.bd;
    return classifier_->BuildGraph(classifier_in, trg, train, cg);
}

template <class SoftmaxOp>
dynet::expr::Expression EncoderClassifier::Forward(const Sentence & sent_src,
                                                 bool train, 
                                                 dynet::ComputationGraph & cg) const {
    if(&cg != curr_graph_)
        THROW_ERROR("Initialized computation graph and passed comptuation graph don't match."); 
    // Perform encoding with each encoder
    dynet::expr::Expression classifier_in = GetEncodedState(sent_src, train, cg);
    return classifier_->Forward<SoftmaxOp>(classifier_in, cg);
}

// Instantiate
template
dynet::expr::Expression EncoderClassifier::Forward<dynet::Softmax>(const Sentence & sent_src, 
                                                               bool train,
                                                               dynet::ComputationGraph & cg) const;
template
dynet::expr::Expression EncoderClassifier::Forward<dynet::LogSoftmax>(const Sentence & sent_src, 
                                                               bool train,
                                                               dynet::ComputationGraph & cg) const;

EncoderClassifier* EncoderClassifier::Read(const DictPtr & vocab_src, const DictPtr & vocab_trg, std::istream & in, dynet::Model & model) {
    int num_encoders;
    string version_id, line;
    if(!getline(in, line))
        THROW_ERROR("Premature end of model file when expecting EncoderClassifier");
    istringstream iss(line);
    iss >> version_id >> num_encoders;
    if(version_id != "enccls_001")
        THROW_ERROR("Expecting a EncoderClassifier of version enccls_001, but got something different:" << endl << line);
    vector<LinearEncoderPtr> encoders;
    while(num_encoders-- > 0)
        encoders.push_back(LinearEncoderPtr(LinearEncoder::Read(in, model)));
    ClassifierPtr classifier(Classifier::Read(in, model));
    return new EncoderClassifier(encoders, classifier, model);
}
void EncoderClassifier::Write(std::ostream & out) {
    out << "enccls_001 " << encoders_.size() << endl;
    for(auto & enc : encoders_) enc->Write(out);
    classifier_->Write(out);
}
