#pragma once

#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>

namespace lamtram {

inline std::vector<std::string> Tokenize(const char *str, char c = ' ') {
    std::vector<std::string> result;
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
