#pragma once

#include <lamtram/sentence.h>
#include <lamtram/dict-utils.h>

#include <unordered_map>
#include <memory>
#include <iostream>

namespace lamtram {

typedef typename std::unordered_map<std::string, std::pair<std::string, float> > UniqueStringMapping;
typedef typename std::shared_ptr<UniqueStringMapping> UniqueStringMappingPtr;

typedef typename std::unordered_map<WordId, std::vector<std::pair<WordId, float> > > MultipleIdMapping;
typedef typename std::shared_ptr<MultipleIdMapping> MultipleIdMappingPtr;

UniqueStringMapping* LoadUniqueStringMapping(std::istream & in);
UniqueStringMapping* LoadUniqueStringMapping(const std::string & filename);

MultipleIdMapping* LoadMultipleIdMapping(std::istream & in, const DictPtr & vocab_src, const DictPtr & vocab_trg);
MultipleIdMapping* LoadMultipleIdMapping(const std::string & filename, const DictPtr & vocab_src, const DictPtr & vocab_trg);

}
