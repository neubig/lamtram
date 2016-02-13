
#include <iostream>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>
#include <boost/range/irange.hpp>
#include <boost/algorithm/string.hpp>
#include <cnn/dict.h>
#include <lamtram/dist-train.h>
#include <lamtram/macros.h>
#include <lamtram/timer.h>
#include <lamtram/sentence.h>
#include <lamtram/dist-base.h>
#include <lamtram/dist-factory.h>
#include <lamtram/dict-utils.h>

using namespace std;
using namespace lamtram;
namespace po = boost::program_options;

int DistTrain::main(int argc, char** argv) {
  po::options_description desc("*** lamtram-train (by Graham Neubig) ***");
  desc.add_options()
    ("help", "Produce help message")
    ("vocab_file", po::value<string>()->default_value(""), "Vocab file")
    ("train_file", po::value<string>()->default_value(""), "Training file")
    ("model_out", po::value<string>()->default_value(""), "File to write the model to")
    ("sig", po::value<string>()->default_value("ngram_lin_1_2_3"), "Signature for the language model")
    ("verbose", po::value<int>()->default_value(0), "How much verbose output to print")
    ;
  boost::program_options::variables_map vm_;
  po::store(po::parse_command_line(argc, argv, desc), vm_);
  po::notify(vm_);   
  if (vm_.count("help")) {
    cout << desc << endl;
    return 1;
  }

  // Open output file
  ofstream model_out(vm_["model_out"].as<string>());
  if(!model_out)
    THROW_ERROR("Could not write to output file: " << vm_["model_out"].as<string>());

  GlobalVars::verbose = vm_["verbose"].as<int>();

  // Read in the vocabulary if necessary
  string line;
  DictPtr dict(new cnn::Dict);
  dict->Convert("<unk>");
  dict->Convert("<s>");
  if(vm_["vocab_file"].as<string>() != "") {
    ifstream vocab_file(vm_["vocab_file"].as<string>());
    if(!(getline(vocab_file, line) && line == "<unk>" && getline(vocab_file, line) && line == "<s>"))
    THROW_ERROR("First two lines of a vocabulary file must be <unk> and <s>");
    while(getline(vocab_file, line))
    dict->Convert(line);
    dict->Freeze();
    dict->SetUnk("<unk>");
  }

  // Create the model
  DistPtr dist(DistFactory::create_dist(vm_["sig"].as<string>()));

  // Read in the data
  {
    ifstream train_file(vm_["train_file"].as<string>());
    if(!train_file) THROW_ERROR("Couldn't open file: " << vm_["train_file"].as<string>());
    while(getline(train_file, line)) {
    dist->add_stats(ParseSentence(line, dict, true));
    }
  }
  dist->finalize_stats();

  // Write the model
  model_out << dist->get_sig() << endl;
  dist->write(dict, model_out);

  return 0;
}
