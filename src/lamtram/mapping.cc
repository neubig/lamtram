
#include <lamtram/mapping.h>
#include <lamtram/macros.h>
#include <dynet/dict.h>
#include <boost/algorithm/string.hpp>
#include <fstream>

using namespace std;

namespace lamtram {

UniqueStringMapping* LoadUniqueStringMapping(const std::string & filename) {
  ifstream map_in(filename);
  if(!map_in)
    THROW_ERROR("Could not find map_in file " << filename);
  return LoadUniqueStringMapping(map_in);
}
  
UniqueStringMapping* LoadUniqueStringMapping(istream & map_in) {
  UniqueStringMapping* ret = new UniqueStringMapping;
  string line;
  vector<string> strs;
  while(getline(map_in, line)) {
    boost::split(strs, line, boost::is_any_of("\t"));
    if(strs.size() != 3)
      THROW_ERROR("Invalid line in mapping file: " << line);
    float my_score = stof(strs[2]);
    auto it = ret->find(strs[0]);
    if(it == ret->end() || it->second.second < my_score)
      (*ret)[strs[0]] = make_pair(strs[1], my_score);
  }
  return ret;
}

MultipleIdMapping* LoadMultipleIdMapping(const std::string & filename, const DictPtr & vocab_src, const DictPtr & vocab_trg) {
  ifstream map_in(filename);
  if(!map_in)
    THROW_ERROR("Could not find map_in file " << filename);
  return LoadMultipleIdMapping(map_in, vocab_src, vocab_trg);
}
  
MultipleIdMapping* LoadMultipleIdMapping(istream & map_in, const DictPtr & vocab_src, const DictPtr & vocab_trg) {
  MultipleIdMapping* ret = new MultipleIdMapping;
  string line;
  vector<string> strs;
  while(getline(map_in, line)) {
    boost::split(strs, line, boost::is_any_of("\t"));
    if(strs.size() != 3)
      THROW_ERROR("Invalid line in mapping file: " << line);
    float my_score = stof(strs[2]);
    (*ret)[vocab_src->convert(strs[0])].push_back(make_pair(vocab_trg->convert(strs[1]), my_score));
  }
  return ret;
}

}
