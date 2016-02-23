#include <fstream>
#include <lamtram/dist-factory.h>
#include <lamtram/dist-ngram.h>
#include <lamtram/dist-uniform.h>
#include <lamtram/dist-unk.h>
#include <lamtram/dist-one-hot.h>
#include <lamtram/sentence.h>
#include <lamtram/macros.h>
#include <lamtram/input-file-stream.h>


using namespace std;
using namespace lamtram;

DistPtr DistFactory::create_dist(const std::string & sig) {
  if(sig.substr(0, 5) == "ngram") {
    return DistPtr(new DistNgram(sig));
  } else if(sig == "uniform") {
    return DistPtr(new DistUniform(sig));
  } else if(sig == "unk") {
    return DistPtr(new DistUnk(sig));
  } else if(sig == "onehot") {
    return DistPtr(new DistOneHot(sig));
  } else {
    THROW_ERROR("Bad distribution signature");
  }
}

DistPtr DistFactory::from_file(const std::string & file_name, DictPtr dict) {
  InputFileStream in(file_name);
  if(!in) THROW_ERROR("Could not open " << file_name);
  string line;
  if(!getline(in, line)) THROW_ERROR("Premature end of file");
  DistPtr ret = DistFactory::create_dist(line);
  ret->read(dict, in);
  return ret;
}
