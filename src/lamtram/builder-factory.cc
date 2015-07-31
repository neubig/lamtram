#include <lamtram/builder-factory.h>
#include <lamtram/macros.h>
#include <cnn/model.h>
#include <cnn/rnn.h>
#include <cnn/lstm.h>
#include <cnn/gru.h>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace lamtram;

BuilderSpec::BuilderSpec(const std::string & spec) {
    vector<string> strs;
    boost::algorithm::split(strs, spec, boost::is_any_of(":"));
    if(strs.size() != 3)
        THROW_ERROR("Invalid layer specification \"" << spec << "\", must be layer type (rnn/lstm/gru), number of nodes, number of layers, with the three elements separated by a token.");
    type = strs[0];
    nodes = boost::lexical_cast<int>(strs[1]); 
    layers = boost::lexical_cast<int>(strs[2]); 
}

BuilderPtr BuilderFactory::CreateBuilder(const BuilderSpec & spec, int input_dim, cnn::Model & model) {
    // cerr << "BuilderFactor::CreateBuilder(" << spec << ", " << input_dim << ", " << (long)&model << ")" << endl;
    if(spec.type == "rnn") {
        return BuilderPtr(new cnn::SimpleRNNBuilder(spec.layers, input_dim, spec.nodes, &model));
    } else if(spec.type == "lstm") {
        return BuilderPtr(new cnn::LSTMBuilder(spec.layers, input_dim, spec.nodes, &model));
    } else if(spec.type == "gru") {
        THROW_ERROR("GRU spec.nodes are not supported yet.");
        // return BuilderPtr(new cnn::GRUBuilder(spec.layers, input_dim, spec.nodes, &model));
    } else {
        THROW_ERROR("Unknown layer type " << spec.type);
    }
}
