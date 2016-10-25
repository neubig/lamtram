#pragma once

#include <memory>
#include <iostream>
#include <vector>
#include <lamtram/sentence.h>

namespace dynet { class Dict; }

namespace lamtram {

typedef std::shared_ptr<dynet::Dict> DictPtr;

std::vector<std::string> SplitWords(const std::string & str);
Sentence ParseWords(dynet::Dict & dict, const std::string & str, bool sent_end);
Sentence ParseWords(dynet::Dict & dict, const std::vector<std::string> & str, bool sent_end);
std::string PrintWords(dynet::Dict & dict, const Sentence & sent);
std::string PrintWords(const std::vector<std::string> & sent);
std::vector<std::string> ConvertWords(dynet::Dict & sd, const Sentence & sent, bool sent_end);
void WriteDict(const dynet::Dict & dict, const std::string & file);
void WriteDict(const dynet::Dict & dict, std::ostream & out);
dynet::Dict* ReadDict(const std::string & file);
dynet::Dict* ReadDict(std::istream & in);
dynet::Dict* CreateNewDict(bool add_symbols = true);

}
