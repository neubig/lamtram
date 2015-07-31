#include <lamtram/vocabulary.h>
#include <lamtram/macros.h>
#include <boost/lexical_cast.hpp>
#include <sstream>
#include <cassert>

using namespace std;
using namespace lamtram;
using namespace boost;

WordId Vocabulary::WID(const string str) {
    auto it = wids_.find(str);
    if(it != wids_.end())
        return it->second;
    if(freeze_)
        return default_;
    WordId new_id = wids_.size();
    wsyms_.push_back(str);
    wids_.insert(make_pair(str, new_id));
    return new_id;
}
string Vocabulary::WSym(const WordId wid) const {
    assert(wid < (WordId)wsyms_.size());
    if(wid < 0) return "<null>";
    return wsyms_[wid];
}

Sentence Vocabulary::ParseWords(const string & str, int pad, bool add_final) {
    char c = ' ';
    Sentence result(pad, 0);
    const char* ptr = &str[0];
    while(1) {
        while(*ptr == c && *ptr)
            ptr++;
        const char *begin = ptr;
        while(*ptr != c && *ptr)
            ptr++;
        if(ptr == begin)
            break;
        result.push_back(WID(string(begin, ptr)));
    }
    if(add_final)
        result.push_back(0);
    return result;
}
string Vocabulary::PrintWords(const Sentence & sent, bool sentend) const {
    ostringstream oss;
    string filler;
    for(WordId wid : sent) {
        if(sentend || wid != 0) {
            oss << filler << WSym(wid);
            filler = " ";
        }
    }
    return oss.str();
}

void Vocabulary::Write(ostream & out) const {
    out << wsyms_.size() << endl;
    for(auto & word : wsyms_)
        out << word << endl;
}

Vocabulary * Vocabulary::Read(istream & in) {
    string line;
    if(!getline(in, line)) THROW_ERROR("Premature end of vocabulary");
    int size = lexical_cast<int>(line);
    Vocabulary * ret = new Vocabulary;
    ret->wsyms_.resize(size);
    for(int i = 0; i < size; i++) {
        if(!getline(in, line)) THROW_ERROR("Premature end of vocabulary");
        ret->wsyms_[i] = line;
        ret->wids_[line] = i;
    }
    return ret;
}
