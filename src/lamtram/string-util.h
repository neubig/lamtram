#pragma once

#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>

namespace lamtram {

inline unsigned int UTF8Len(const unsigned char x) {
  if (x < 0x80) return 1;
  else if ((x >> 5) == 0x06) return 2;
  else if ((x >> 4) == 0x0e) return 3;
  else if ((x >> 3) == 0x1e) return 4;
  else if ((x >> 2) == 0x3e) return 5;
  else if ((x >> 1) == 0x7e) return 6;
  else return 0;
}

inline std::vector<std::string> TokenizeWildcarded(const std::string & str, const std::vector<std::string> & wildcards, const std::string & delimiter) {
  std::vector<std::string> ret;
  auto end = str.find("WILD");
  if(end != std::string::npos) {
    std::string left = str.substr(0, end);
    std::string right = str.substr(end+4);
    for(size_t i = 0; i < wildcards.size(); i++)
      ret.push_back(left+wildcards[i]+right);
  } else {
    boost::split(ret, str, boost::is_any_of(delimiter));
  }
  return ret;
}


inline std::vector<std::string> Tokenize(const char *str, char c = ' ') {  
  std::vector<std::string> result;
  if(*str == 0) return result;
  while(1) {
    const char *begin = str;
    while(*str != c && *str)
      str++;
    result.push_back(std::string(begin, str));
    if(0 == *str++)
      break;
  }
  return result;
}
inline std::vector<std::string> Tokenize(const std::string &str, char c = ' ') {
  return Tokenize(str.c_str(), c);
}
inline std::vector<std::string> Tokenize(const std::string & str, const std::string & delim) {
  std::vector<std::string> vec;
  if(str == "") return vec;
  size_t loc, prev = 0;
  while((loc = str.find(delim, prev)) != std::string::npos) {
    vec.push_back(str.substr(prev, loc-prev));
    prev = loc + delim.length();
  }
  vec.push_back(str.substr(prev, str.size()-prev));
  return vec;
}
inline std::string FirstToken(const std::string & str, char c = ' ') {
  const char *end = &str[0];
  while(*end != 0 && *end != c)
    end++;
  return std::string(&str[0], end);
}

inline std::string EscapeQuotes(std::string ret) {
  boost::replace_all(ret, "\\", "\\\\");
  boost::replace_all(ret, "\"", "\\\"");
  return ret;
}

}  // end namespace
