#pragma once

#include <cassert>
#include <iostream>
#include <string>
#include <memory>
#include <tuple>
#include <cnpy/cnpy.h>
#include <cnn/model.h>
#include <cnn/rnn.h>
#include <cnn/tensor.h>
#include <lamtram/builder-factory.h>

namespace cnn {
struct LookupParameter;
struct RNNBuilder;
}

namespace lamtram {

class CnpyUtils {
public:
    static void copyWeight(const std::string & name,cnpy::npz_t & model,cnn::LookupParameter & target);
    static void copyGRUWeight(const std::string & prefix,cnpy::npz_t & model,BuilderPtr target);
    static void copyGRUCondWeight(const std::string & prefix,cnpy::npz_t & model,BuilderPtr target);
    static std::pair<int,int> getData(const std::string & name, cnpy::npz_t & model, float * & data);
    static void splitData(const float * data, std::vector<float> & f1, std::vector<float> & f2, const std::pair<int,int> & size);
};

}
