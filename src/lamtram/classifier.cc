#include <lamtram/classifier.h>
#include <lamtram/macros.h>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <cnn/nodes.h>

using namespace lamtram;
using namespace std;
using namespace cnn::expr;


Classifier::Classifier(int input_size, int label_size,
                       const std::string & layers, cnn::Model & mod) :
        input_size_(input_size), label_size_(label_size), layer_str_(layers), curr_graph_(NULL) {
    vector<string> layer_sizes;
    if(layers.size() > 0)
        boost::split(layer_sizes, layers, boost::is_any_of(":"));
    layer_sizes.push_back(boost::lexical_cast<string>(label_size));
    int last_size = input_size;
    for(auto & my_lay : layer_sizes) {
        int my_size = boost::lexical_cast<int>(my_lay);    
        p_W_.push_back(mod.add_parameters({(unsigned int)my_size, (unsigned int)last_size}));
        p_b_.push_back(mod.add_parameters({(unsigned int)my_size}));
        last_size = my_size;
    }
    i_W_.resize(p_W_.size());
    i_b_.resize(p_b_.size());
}

cnn::expr::Expression Classifier::BuildGraph(const cnn::expr::Expression & input, int label,
                                             cnn::ComputationGraph & cg) const {
    if(&cg != curr_graph_)
        THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
    cnn::expr::Expression last_expr = input, aff;
    int i;
    for(i = 0; i < (int)p_W_.size()-1; i++) {
        aff = affine_transform({i_b_[i], i_W_[i], last_expr});
        last_expr = tanh({aff});
    }
    aff = affine_transform({i_b_[i], i_W_[i], last_expr});
    return pickneglogsoftmax({aff}, label);
}


template <class SoftmaxOp>
cnn::expr::Expression Classifier::Forward(const cnn::expr::Expression & input,
                                          cnn::ComputationGraph & cg) const {
    if(&cg != curr_graph_)
        THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
    cnn::expr::Expression last_expr = input, aff;
    int i;
    for(i = 0; i < (int)p_W_.size()-1; i++) {
        aff = affine_transform({i_b_[i], i_W_[i], last_expr});
        last_expr = tanh({aff});
    }
    aff = affine_transform({i_b_[i], i_W_[i], last_expr});
    return cnn::expr::Expression(aff.pg, aff.pg->add_function<SoftmaxOp>({aff.i}));
}

// Instantiate
template
cnn::expr::Expression Classifier::Forward<cnn::Softmax>(
                                     const cnn::expr::Expression & input,
                                     cnn::ComputationGraph & cg) const;
template
cnn::expr::Expression Classifier::Forward<cnn::LogSoftmax>(
                                     const cnn::expr::Expression & input,
                                     cnn::ComputationGraph & cg) const;

void Classifier::NewGraph(cnn::ComputationGraph & cg) {
    for(int i = 0; i < (int)p_W_.size(); i++) {
        i_b_[i] = parameter(cg, p_b_[i]);
        i_W_[i] = parameter(cg, p_W_[i]);
    }
    curr_graph_ = &cg;
}


Classifier* Classifier::Read(std::istream & in, cnn::Model & model) {
    int input_size, label_size;
    string version_id, layers, line;
    if(!getline(in, line))
        THROW_ERROR("Premature end of model file when expecting Classifier");
    istringstream iss(line);
    iss >> version_id >> input_size >> label_size >> layers;
    if(version_id != "cls_001") {
        THROW_ERROR("Expecting a Classifier of version cls_001, but got something different:" << endl << line);
    }
    return new Classifier(input_size, label_size, layers, model);
}
void Classifier::Write(std::ostream & out) {
    out << "cls_001 " << input_size_ << " " << label_size_ << " " << layer_str_ << endl;
}
