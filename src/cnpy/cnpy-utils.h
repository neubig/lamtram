#pragma once

#include <cassert>
#include <iostream>
#include <string>
#include <memory>
#include <tuple>
#include <cnpy/cnpy.h>
#include <dynet/model.h>
#include <dynet/rnn.h>
#include <dynet/tensor.h>
#include <lamtram/builder-factory.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/softmax-base.h>
#include <lamtram/softmax-full.h>
#include <lamtram/softmax-multilayer.h>

namespace dynet {
struct LookupParameter;
struct RNNBuilder;
}

namespace lamtram {
class ExternAttentional;
typedef std::shared_ptr<ExternAttentional> ExternAttentionalPtr;

class CnpyUtils {
public:
    static void copyWeight(const std::string & name,cnpy::npz_t & model,dynet::LookupParameter & target,float dropoutProb);
    static void copyGRUWeight(const std::string & prefix,cnpy::npz_t & model,BuilderPtr target);
    static void copyGRUCondWeight(const std::string & prefix,cnpy::npz_t & model,BuilderPtr target);
    static void copyAttentionWeight(const std::string & prefix,cnpy::npz_t & model,ExternAttentionalPtr target);
    static void copySoftmaxWeight(const std::string & prefix,cnpy::npz_t & model,SoftmaxPtr target,int vocSize);
    static std::pair<int,int> getData(const std::string & name, cnpy::npz_t & model, float * & data);
    static void splitData(const float * data, std::vector<float> & f1, std::vector<float> & f2, const std::pair<int,int> & size);
};

}
