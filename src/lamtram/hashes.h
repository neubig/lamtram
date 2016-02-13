#pragma once

// TODO: Could probably do much better here
namespace std {
  template <class T> struct hash<std::vector<T> > {
    size_t operator()(const std::vector<T>  & x) const
    {
      size_t hash = 5381;
      const char* c = (const char*)&x[0];
      const char* end = (const char*)&x[x.size()];
      while(c != end)
        hash = ((hash << 5) + hash) + *(c++);
      return hash;
    }
  };
  template <class T1, class T2> struct hash<std::pair<T1,T2> > {
    size_t operator()(const std::pair<T1,T2> & x) const
    {
      return (std::hash<T1>()(x.first) << 16) ^ std::hash<T2>()(x.second);
    }
  };
}
