#pragma once

#include <memory>
#include <lamtram/sentence.h>

namespace cnn { class Dict; }

namespace lamtram {

typedef std::shared_ptr<cnn::Dict> DictPtr;
Sentence ParseSentence(const std::string & str, DictPtr dict, bool sent_end);
std::string PrintSentence(const Sentence & sent, DictPtr dict);

}
