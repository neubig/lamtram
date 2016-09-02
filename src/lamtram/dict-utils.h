#pragma once

#include <memory>
#include <iostream>
#include <vector>
#include <lamtram/sentence.h>

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

namespace cnn { class Dict; }

namespace lamtram {

typedef std::shared_ptr<cnn::Dict> DictPtr;

std::vector<std::string> SplitWords(const std::string & str);
Sentence ParseWords(cnn::Dict & dict, const std::string & str, bool sent_end);
Sentence ParseWords(cnn::Dict & dict, const std::vector<std::string> & str, bool sent_end);
std::string PrintWords(cnn::Dict & dict, const Sentence & sent);
std::string PrintWords(const std::vector<std::string> & sent);
std::vector<std::string> ConvertWords(cnn::Dict & sd, const Sentence & sent, bool sent_end);
void WriteDict(const cnn::Dict & dict, const std::string & file);
void WriteDict(const cnn::Dict & dict, std::ostream & out);
cnn::Dict* ReadDict(const std::string & file);
cnn::Dict* ConvertDict(const std::string & file,int size);
cnn::Dict* ReadDict(std::istream & in);
cnn::Dict* CreateNewDict(bool add_symbols = true);

}
