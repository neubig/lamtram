#include <iostream>
#include <fstream>
#include <sstream>
#include <dynet/dict.h>
#include <lamtram/dict-utils.h>
#include <lamtram/macros.h>

using namespace std;

namespace lamtram {

vector<string> SplitWords(const std::string & line) {
  std::istringstream in(line);
  std::string word;
  std::vector<std::string> res;
  while(in) {
    in >> word;
    if (!in || word.empty()) break;
    res.push_back(word);
  }
  return res;
}

Sentence ParseWords(dynet::Dict & sd, const std::string& line, bool add_end) {
  std::istringstream in(line);
  std::string word;
  std::vector<int> res;
  while(in) {
    in >> word;
    if (!in || word.empty()) break;
    res.push_back(sd.convert(word));
  }
  if(add_end && (res.size() == 0 || *res.rbegin() != 0))
    res.push_back(0);
  return res;
}

Sentence ParseWords(dynet::Dict & sd, const std::vector<std::string>& line, bool add_end) {
  Sentence res;
  for(const std::string & word : line)
    res.push_back(sd.convert(word));
  if(add_end && (res.size() == 0 || *res.rbegin() != 0))
    res.push_back(0);
  return res;
}

std::string PrintWords(dynet::Dict & sd, const Sentence & sent) {
  ostringstream oss;
  if(sent.size())
    oss << sd.convert(sent[0]);
  for(size_t i = 1; i < sent.size(); i++)
    oss << ' ' << sd.convert(sent[i]);
  return oss.str();
}
std::string PrintWords(const std::vector<std::string> & sent) {
  ostringstream oss;
  if(sent.size())
    oss << sent[0];
  for(size_t i = 1; i < sent.size(); i++)
    oss << ' ' << sent[i];
  return oss.str();
}

vector<string> ConvertWords(dynet::Dict & sd, const Sentence & sent, bool sentend) {
  vector<string> ret;
  for(WordId wid : sent) {
    if(sentend || wid != 0)
      ret.push_back(sd.convert(wid));
  }
  return ret;
}

void WriteDict(const dynet::Dict & dict, const std::string & file) {
  ofstream out(file);
  if(!out) THROW_ERROR("Could not open file: " << file);
  WriteDict(dict, out);
}
void WriteDict(const dynet::Dict & dict, std::ostream & out) {
  out << "dict_v001" << '\n';
  for(const auto & str : dict.get_words())
    out << str << '\n';
  out << endl;
}
dynet::Dict* ReadDict(const std::string & file) {
  ifstream in(file);
  if(!in) THROW_ERROR("Could not open file: " << file);
  return ReadDict(in);
}
dynet::Dict* ReadDict(std::istream & in) {
  dynet::Dict* dict = new dynet::Dict;
  string line;
  if(!getline(in, line) || line != "dict_v001")
    THROW_ERROR("Expecting dictionary version dict_v001, but got: " << line);
  while(getline(in, line)) {
    if(line == "") break;
    dict->convert(line);
  }
  bool has_unk = dict->convert("<unk>") == 1;
  dict->freeze();
  if(has_unk)
    dict->set_unk("<unk>");
  return dict;
}
dynet::Dict * CreateNewDict(bool add_tokens) {
  dynet::Dict * ret = new dynet::Dict;
  if(add_tokens) {
    ret->convert("<s>");
    ret->convert("<unk>");
  }
  return ret;
}

}
