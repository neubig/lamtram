#pragma once

#include <lamtram/sentence.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>

namespace lamtram {

class Vocabulary {
public:

  typedef std::unordered_map<std::string,WordId> WordMap;

  Vocabulary(std::string sent_end = "<s>") 
      : wids_(), wsyms_(), freeze_(false), default_(-1) {
    if(sent_end != "") {
      wids_[sent_end] = 0;
      wsyms_.push_back(sent_end);
    }
  }
  ~Vocabulary() { }

  WordId WID(const std::string str);
  std::string WSym(const WordId wid) const;

  static std::vector<std::string> SplitWords(const std::string & str);
  Sentence ParseWords(const std::string & str, int pad = 0, bool add_final = false);
  Sentence ParseWords(const std::vector<std::string> & str, int pad = 0, bool add_final = false);
  std::vector<std::string> ConvertWords(const Sentence & sent, bool sentend = true) const;
  std::string PrintWords(const Sentence & sent, bool sentend = true) const;
  std::string PrintWords(const std::vector<std::string> & sent) const;

  WordId GetDefault() const { return default_; }
  void SetDefault(const std::string & str) {
    default_ = WID(str);
  }
  void SetFreeze(bool freeze) { freeze_ = freeze; }

  const WordMap & GetWIDs() const { return wids_; }
  const std::vector<std::string> & GetWSyms() const { return wsyms_; }
  WordMap & GetWIDs() { return wids_; }
  std::vector<std::string> & GetWSyms() { return wsyms_; }

  size_t size() const { return wsyms_.size(); }

  void Write(std::ostream & out) const;
  static Vocabulary * Read(std::istream & in);

  bool operator==(const Vocabulary & rhs) {
    return wids_ == rhs.wids_ && wsyms_ == rhs.wsyms_ && default_ == rhs.default_;
  }
  bool operator!=(const Vocabulary & rhs) {
    return !(*this == rhs);
  }

protected:
  WordMap wids_;
  std::vector<std::string> wsyms_;
  bool freeze_;
  WordId default_;
};

typedef std::shared_ptr<Vocabulary> VocabularyPtr;


}
