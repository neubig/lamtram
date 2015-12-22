#pragma once

#include <vector>
#include <unordered_map>

namespace lamtram {
typedef int WordId;
typedef std::vector<WordId> Sentence;
typedef std::vector<std::vector<float> > Alignment;
typedef std::unordered_map<std::string, std::pair<std::string, float> > Mapping;
}
