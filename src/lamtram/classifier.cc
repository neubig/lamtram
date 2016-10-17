#include <lamtram/classifier.h>
#include <lamtram/macros.h>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <dynet/nodes.h>

using namespace lamtram;
using namespace std;
using namespace dynet::expr;


Classifier::Classifier(int input_size, int label_size,
             const std::string & layers, const std::string & smsig, dynet::Model & mod) :
    input_size_(input_size), label_size_(label_size), layer_str_(layers), smsig_(smsig), curr_graph_(NULL), dropout_(0.f) {
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

namespace lamtram {

template <>
dynet::expr::Expression Classifier::BuildGraph<int>(const dynet::expr::Expression & input, const int & label,
                          bool train,
                          dynet::ComputationGraph & cg) const {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
  dynet::expr::Expression last_expr = input, aff;
  int i;
  for(i = 0; i < (int)p_W_.size()-1; i++) {
    aff = affine_transform({i_b_[i], i_W_[i], last_expr});
    last_expr = tanh({aff});
    if(dropout_) last_expr = dropout(last_expr, dropout_);
  }
  aff = affine_transform({i_b_[i], i_W_[i], last_expr});
  if(smsig_ == "full") {
    return pickneglogsoftmax({aff}, label);
  } else if(smsig_ == "hinge") {
    return hinge({aff}, label);
  } else {
    THROW_ERROR("Illegal softmax signature: " << smsig_);
  }
}

template <>
dynet::expr::Expression Classifier::BuildGraph<vector<int> >(const dynet::expr::Expression & input, const vector<int> & label,
                               bool train,
                               dynet::ComputationGraph & cg) const {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
  dynet::expr::Expression last_expr = input, aff;
  int i;
  for(i = 0; i < (int)p_W_.size()-1; i++) {
    aff = affine_transform({i_b_[i], i_W_[i], last_expr});
    last_expr = tanh({aff});
    if(dropout_) last_expr = dropout(last_expr, dropout_);
  }
  aff = affine_transform({i_b_[i], i_W_[i], last_expr});
  vector<unsigned> un_label(label.size());
  for(i = 0; i < (int)un_label.size(); i++)
    un_label[i] = label[i];
  if(smsig_ == "full") {
    return sum_batches(pickneglogsoftmax({aff}, un_label));
  } else if(smsig_ == "hinge") {
    return sum_batches(hinge({aff}, un_label));
  } else {
    THROW_ERROR("Illegal softmax signature: " << smsig_);
  }
}

}


template <class SoftmaxOp>
dynet::expr::Expression Classifier::Forward(const dynet::expr::Expression & input,
                      dynet::ComputationGraph & cg) const {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed comptuation graph don't match.");
  dynet::expr::Expression last_expr = input, aff;
  int i;
  for(i = 0; i < (int)p_W_.size()-1; i++) {
    aff = affine_transform({i_b_[i], i_W_[i], last_expr});
    last_expr = tanh({aff});
    if(dropout_) last_expr = dropout(last_expr, dropout_);
  }
  aff = affine_transform({i_b_[i], i_W_[i], last_expr});
  return dynet::expr::Expression(aff.pg, aff.pg->add_function<SoftmaxOp>({aff.i}));
}

// Instantiate
template
dynet::expr::Expression Classifier::Forward<dynet::Softmax>(
                   const dynet::expr::Expression & input,
                   dynet::ComputationGraph & cg) const;
template
dynet::expr::Expression Classifier::Forward<dynet::LogSoftmax>(
                   const dynet::expr::Expression & input,
                   dynet::ComputationGraph & cg) const;
template
dynet::expr::Expression Classifier::Forward<dynet::Identity>(
                   const dynet::expr::Expression & input,
                   dynet::ComputationGraph & cg) const;

void Classifier::NewGraph(dynet::ComputationGraph & cg) {
  for(int i = 0; i < (int)p_W_.size(); i++) {
    i_b_[i] = parameter(cg, p_b_[i]);
    i_W_[i] = parameter(cg, p_W_[i]);
  }
  curr_graph_ = &cg;
}


Classifier* Classifier::Read(std::istream & in, dynet::Model & model) {
  int input_size, label_size;
  string version_id, layers, line, smsig;
  if(!getline(in, line))
    THROW_ERROR("Premature end of model file when expecting Classifier");
  istringstream iss(line);
  if(version_id != "cls_002") {
    iss >> version_id >> input_size >> label_size >> layers;
    smsig = "full";
  } else if(version_id == "cls_002") {
    iss >> version_id >> input_size >> label_size >> layers >> smsig;
  } else {
    THROW_ERROR("Expecting a Classifier of version cls_001 or cls_002, but got something different:" << endl << line);
  }
  return new Classifier(input_size, label_size, layers, smsig, model);
}
void Classifier::Write(std::ostream & out) {
  out << "cls_001 " << input_size_ << " " << label_size_ << " " << layer_str_ << endl;
}
