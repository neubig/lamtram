#pragma once

#include <boost/program_options.hpp>

namespace lamtram {

class Cnntrans {

public:
    Cnntrans() { }
    int main(int argc, char** argv);

    int SequenceOperation(const boost::program_options::variables_map & vm);
    int ClassifierOperation(const boost::program_options::variables_map & vm);

protected:
    boost::program_options::variables_map vm_;

};

}
