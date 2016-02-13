#include <iostream>
#include <sstream>
#include <cnn/dict.h>
#include <lamtram/dict-utils.h>

using namespace std;

namespace lamtram {

Sentence ParseSentence(const std::string& line, DictPtr sd, bool add_end) {
  std::istringstream in(line);
  std::string word;
  std::vector<int> res;
  while(in) {
    in >> word;
    if (!in || word.empty()) break;
    res.push_back(sd->Convert(word));
  }
  if(add_end && (res.size() == 0 || *res.rbegin() != 1))
    res.push_back(1);
  return res;
}

std::string PrintSentence(const Sentence & sent, DictPtr sd) {
  ostringstream oss;
  if(sent.size())
    oss << sd->Convert(sent[0]);
  for(size_t i = 1; i < sent.size(); i++)
    oss << ' ' << sd->Convert(sent[i]);
  return oss.str();
}

}
