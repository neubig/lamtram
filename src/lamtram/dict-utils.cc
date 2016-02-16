#include <iostream>
#include <fstream>
#include <sstream>
#include <cnn/dict.h>
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

Sentence ParseWords(cnn::Dict & sd, const std::string& line, bool add_end) {
  std::istringstream in(line);
  std::string word;
  std::vector<int> res;
  while(in) {
    in >> word;
    if (!in || word.empty()) break;
    res.push_back(sd.Convert(word));
  }
  if(add_end && (res.size() == 0 || *res.rbegin() != 0))
    res.push_back(0);
  return res;
}

Sentence ParseWords(cnn::Dict & sd, const std::vector<std::string>& line, bool add_end) {
  Sentence res;
  for(const std::string & word : line)
    res.push_back(sd.Convert(word));
  if(add_end && (res.size() == 0 || *res.rbegin() != 0))
    res.push_back(0);
  return res;
}

std::string PrintWords(cnn::Dict & sd, const Sentence & sent) {
  ostringstream oss;
  if(sent.size())
    oss << sd.Convert(sent[0]);
  for(size_t i = 1; i < sent.size(); i++)
    oss << ' ' << sd.Convert(sent[i]);
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

vector<string> ConvertWords(cnn::Dict & sd, const Sentence & sent, bool sentend) {
  vector<string> ret;
  for(WordId wid : sent) {
    if(sentend || wid != 0)
      ret.push_back(sd.Convert(wid));
  }
  return ret;
}

void WriteDict(const cnn::Dict & dict, const std::string & file) {
  ofstream out(file);
  if(!out) THROW_ERROR("Could not open file: " << file);
  WriteDict(dict, out);
}
void WriteDict(const cnn::Dict & dict, std::ostream & out) {
  out << "dict_v001" << '\n';
  for(const auto & str : dict.GetWords())
    out << str << '\n';
  out << endl;
}
cnn::Dict* ReadDict(const std::string & file) {
  ifstream in(file);
  if(!in) THROW_ERROR("Could not open file: " << file);
  return ReadDict(in);
}
cnn::Dict* ReadDict(std::istream & in) {
  cnn::Dict* dict = new cnn::Dict;
  string line;
  if(!getline(in, line) || line != "dict_v001")
    THROW_ERROR("Expecting dictionary version dict_v001, but got: " << line);
  while(getline(in, line)) {
    if(line == "") break;
    dict->Convert(line);
  }
  dict->Freeze();
  dict->SetUnk("<unk>");
  return dict;
}

}
