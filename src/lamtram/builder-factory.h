#pragma once

#include <iostream>
#include <string>
#include <memory>

namespace cnn {
class Model;
struct RNNBuilder;
}

namespace lamtram {

class ExternCalculator;
typedef std::shared_ptr<ExternCalculator> ExternCalculatorPtr;


class BuilderSpec {
public:
    BuilderSpec(const std::string & str);
    std::string type;
    int nodes, layers, multiplier;
};
inline std::ostream & operator<<(std::ostream & out, const BuilderSpec & spec) {
    return out << spec.type << ":" << spec.nodes << ":" << spec.layers;
}

typedef std::shared_ptr<cnn::RNNBuilder> BuilderPtr;
class BuilderFactory {
public:
    static BuilderPtr CreateBuilder(const BuilderSpec & spec, int input_dim, cnn::Model & model);
    static BuilderPtr CreateBuilder(const BuilderSpec & spec, int input_dim, cnn::Model & model,ExternCalculatorPtr & att);
};

}
