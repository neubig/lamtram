#pragma once

#include <iostream>
#include <string>
#include <memory>

namespace dynet {
class Model;
struct RNNBuilder;
}

namespace lamtram {

class ExternCalculator;
class RNNCONDBuilder;
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

typedef std::shared_ptr<dynet::RNNBuilder> BuilderPtr;
typedef std::shared_ptr<RNNCONDBuilder> CondBuilderPtr;
class BuilderFactory {
public:
    static BuilderPtr CreateBuilder(const BuilderSpec & spec, int input_dim, dynet::Model & model);
    static CondBuilderPtr CreateBuilder(const BuilderSpec & spec, int input_dim, int input2_dim, dynet::Model & model,ExternCalculatorPtr & att);
};

}
