#include <lamtram/eval-measure-char.h>

#include <lamtram/eval-measure-loader.h>
#include <lamtram/macros.h>

#include <string-util.h>

#include <dynet/dict.h>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace lamtram;
using namespace boost;


Sentence EvalMeasureChar::ConvertSentence(const Sentence & sentence,std::map<std::string,int> & dict) const {
    
    stringstream s;
    
    
    for(int i =0; i < sentence.size(); i++) {
        if(sentence[i] != EOS)
            s << vocab_.convert(sentence[i]) << " ";
    }

    string sent = s.str();
    boost::algorithm::trim(sent);



    if(ignore_bpe_) {
        boost::replace_all(sent, "@@ ", "");
    }
    
    unsigned pos = 0;
    unsigned len = 0;
    unsigned count = 0;
    
    vector<int> result;
    
    while(pos < sent.size()) {
        len = UTF8Len(sent[pos]);
        string c = sent.substr(pos,len);
        if(use_space_ || c.compare(" ") != 0) {
            if(dict.find(c) == dict.end()) {
                dict[c] = dict.size();
            }
            result.push_back(dict[c]);
        }
        pos += len;
    }

    return result;
}

// Measure the score of the sys output according to the ref
EvalStatsPtr EvalMeasureChar::CalculateStats(const Sentence & ref, const Sentence & sys) const {
    std::map<std::string,int> voc;
    Sentence charRef = ConvertSentence(ref,voc);
    Sentence charSys = ConvertSentence(sys,voc);
    return measure_->CalculateStats(charRef,charSys);
}

EvalStatsPtr EvalMeasureChar::CalculateCachedStats(
            const std::vector<Sentence> & refs, const std::vector<Sentence> & syss, int ref_cache_id, int sys_cache_id) {

    std::vector<Sentence> charRefs;
    std::vector<Sentence> charSyss;
    for(int i = 0; i < refs.size(); i++) {
        charRefs.push_back(ConvertSentence(refs[i],voc_));
    }

    for(int i = 0; i < syss.size(); i++) {
        charSyss.push_back(ConvertSentence(syss[i],voc_));
    }


    return measure_->CalculateCachedStats(charRefs,charSyss,ref_cache_id,sys_cache_id);           
}

// Read in the stats
EvalStatsPtr EvalMeasureChar::ReadStats(const std::string & line) {
    return measure_->ReadStats(line);
}

EvalMeasureChar::EvalMeasureChar(const std::string & config, const dynet::Dict & vocab) : vocab_(vocab){
    vector<string> arr1, arr2;
    boost::split ( arr1, config, boost::is_any_of(","));
    string measure = "";
    for(int i = 0; i < arr1.size(); i++) {
        boost::split ( arr2, arr1[i], boost::is_any_of("="));
        if(arr2.size() < 2)
            THROW_ERROR("Bad evaluation measure config:" << config);
        if(arr2[0].compare("measure") == 0) {
            if(i != arr1.size() -1 ) {
                vector<string> options(arr1.begin()+i+1,arr1.end());
                measure = boost::join(vector<string>(arr2.begin()+1,arr2.end()),"=")+","+boost::join(options,",");
            }else {
                measure = boost::join(vector<string>(arr2.begin()+1,arr2.end()),"=");
            }
            i=arr1.size();
        }else if(arr2[0].compare("use_space") == 0) {
            use_space_ = boost::lexical_cast<bool>(arr2[1]);
        }else if(arr2[0].compare("ignore_bpe") == 0) {
            ignore_bpe_ = boost::lexical_cast<bool>(arr2[1]);
        }
    }

    if(measure.compare("") == 0) {
            THROW_ERROR("No measure given for char-based measure:" << config);
        
    }
    measure_ = std::shared_ptr<EvalMeasure>(EvalMeasureLoader::CreateMeasureFromString(measure, vocab));
}

