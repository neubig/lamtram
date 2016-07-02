#pragma once

#include <boost/program_options.hpp>
#include <lamtram/sentence.h>
#include <lamtram/mapping.h>

namespace lamtram {

class Lamtram {


public:

  void MapWords(const std::vector<std::string> & src_strs, const Sentence & trg_sent, const Sentence & align, const UniqueStringMappingPtr & mapping, std::vector<std::string> & trg_strs);

  Lamtram() { }
  int main(int argc, char** argv);

  int SequenceOperation(const boost::program_options::variables_map & vm);
  int ClassifierOperation(const boost::program_options::variables_map & vm);

protected:
  boost::program_options::variables_map vm_;

};

}
