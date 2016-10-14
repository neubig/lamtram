#pragma once

#include <sstream>
#include <stdexcept>
#include <iostream>
#include <vector>

namespace lamtram {

class GlobalVars { 
public:
    static int verbose;
    static int curr_word;
    static int layer_size;
};

}

#define THROW_ERROR(msg) do {                   \
    std::ostringstream oss;                     \
    oss << "ERROR: " << msg;                    \
    throw std::runtime_error(oss.str()); }      \
  while (0);

#define CHECK_CLOSE_COLLECTIONS(aa, bb, tolerance) { \
    using std::distance; \
    using std::begin; \
    using std::end; \
    auto a = begin(aa), ae = end(aa); \
    auto b = begin(bb); \
    BOOST_REQUIRE_EQUAL(distance(a, ae), distance(b, end(bb))); \
    for(; a != ae; ++a, ++b) { \
        BOOST_CHECK_CLOSE(*a, *b, tolerance); \
    } \
}

namespace std {

template<class T, class U>
inline std::ostream& operator<<(std::ostream & out, const std::pair<T,U> & a) {
    out << '<' << a.first << ',' << a.second << '>';
    return out;
}

template<class T, class U>
inline std::pair<T,U> & operator+=(std::pair<T,U> & p1, const std::pair<T,U> & p2) {
    p1.first += p2.first;
    p1.second += p2.second;
    return p1;
}

template<class T>
inline std::ostream& operator<<(std::ostream & out, const std::vector<T> & a) {
    if(a.size() == 0) { out << "[]"; }
    else {
        out << '[' << a[0];
        for(unsigned i = 1; i < a.size(); i++) out << ',' << a[i];
        out << ']'; 
    }
    return out;
}

}
