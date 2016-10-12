#include <lamtram/eval-measure-charf.h>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <lamtram/macros.h>
#include <sstream>
#include <string-util.h>
using namespace std;
using namespace lamtram;
using namespace boost;



EvalMeasureCharF::NgramStats * EvalMeasureCharF::ExtractNgrams(const Sentence & sentence) const {
    EvalMeasureCharF::NgramStats * all_ngrams = new EvalMeasureCharF::NgramStats;
    
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
    vector<string> current_ngram;
    unsigned pos = 0;
    unsigned len = 0;
    unsigned count = 0;





    while(current_ngram.size() < ngram_order_ && pos < sent.size()) {
        len = UTF8Len(sent[pos]);
        string c = sent.substr(pos,len);
        if(use_space_ || c.compare(" ") != 0) {
            current_ngram.push_back(c);
        }
        pos += len;
    }

    if(current_ngram.size() == ngram_order_) {
        ++((*all_ngrams)[boost::algorithm::join(current_ngram, " ")]);
        ++count;
    }

    
    while(pos < sent.size()) {
        len = UTF8Len(sent[pos]);
        string c = sent.substr(pos,len);
        if(use_space_ || c.compare(" ") != 0) {
            current_ngram.erase(current_ngram.begin());
            current_ngram.push_back(c);
            ++((*all_ngrams)[boost::algorithm::join(current_ngram, "")]);
            ++count;
        }
        pos += len;
    }

    (*all_ngrams)["<ALLCOUNT>"] = count;

    return all_ngrams;
}

std::shared_ptr<EvalMeasureCharF::NgramStats> EvalMeasureCharF::GetCachedStats(const Sentence & sent, int cache_id) {
    if(cache_id == INT_MAX) return std::shared_ptr<NgramStats>(ExtractNgrams(sent));
    StatsCache::const_iterator it = cache_.find(cache_id);
    if(it == cache_.end()) {
        std::shared_ptr<NgramStats> new_stats(ExtractNgrams(sent));
        cache_.insert(make_pair(cache_id, new_stats));
        return new_stats;
    } else {
        return it->second;
    }
}

std::shared_ptr<EvalStats> EvalMeasureCharF::CalculateStats(const Sentence & ref, const Sentence & sys) const {
    std::shared_ptr<NgramStats> ref_s(ExtractNgrams(ref)), sys_s(ExtractNgrams(sys));
    return CalculateStats(*ref_s, ref.size(), *sys_s, sys.size());
}

// Measure the score of the sys output according to the ref
std::shared_ptr<EvalStats> EvalMeasureCharF::CalculateStats(const Sentence & ref, const Sentence & sys, int ref_cache_id, int sys_cache_id) {
    return CalculateStats(*GetCachedStats(ref, ref_cache_id), ref.size(), *GetCachedStats(sys, sys_cache_id), sys.size());
}


std::shared_ptr<EvalStats> EvalMeasureCharF::CalculateStats(const NgramStats & ref_ngrams, int ref_len,
                                                      const NgramStats & sys_ngrams, int sys_len) const {
    int vals_n = 3;
    vector<EvalStatsDataType> vals(vals_n);

    vals[0] = 0;
    vals[1] = sys_ngrams.find("<ALLCOUNT>")->second;
    vals[2] = ref_ngrams.find("<ALLCOUNT>")->second;
    

    for (NgramStats::const_iterator it = sys_ngrams.begin(); it != sys_ngrams.end(); it++) {
        if(it->first.compare("<ALLCOUNT>") == 0) {
            NgramStats::const_iterator ref_it = ref_ngrams.find(it->first);
            if(ref_it != ref_ngrams.end()) {
                vals[0] += min(ref_it->second,it->second);
            }
        }
    }
    

    // Create the stats for this sentence
    EvalStatsPtr ret(new EvalStatsCharF(vals,smooth_val_, beta_,mode_));
    // If we are using sentence based, take the average immediately
    return ret;
}


// Read in the stats
std::shared_ptr<EvalStats> EvalMeasureCharF::ReadStats(const std::string & line) {
    EvalStatsPtr ret;
    ret.reset(new EvalStatsCharF(std::vector<EvalStatsDataType>(), smooth_val_,beta_,mode_));
    ret->ReadStats(line);
    return ret;
}

std::string EvalStatsCharF::ConvertToString() const {
    CharFReport report = CalcCharFReport();
    ostringstream oss;
    oss << GetIdString() << " = " << report.charF;
    oss << " (Precision=" << report.precision << ", recall=" << report.recall << ", hyp_len=" << report.sys_len << ", ref_len=" << report.ref_len << ")";
    return oss.str();
}



float EvalStatsCharF::ConvertToScore() const {
    return CalcCharFReport().charF;
}


CharFReport EvalStatsCharF::CalcCharFReport() const {
    CharFReport report;
    report.precision = 1.0 * (vals_[0]+ smooth_) / (vals_[2] + smooth_);
    report.recall = 1.0 * (vals_[0] + smooth_) / (vals_[1] + smooth_);
    if(mode_ == 0) {
        report.charF = 1.0 *(1+beta_*beta_) * (report.precision * report.recall) /(beta_*beta_ * report.precision + report.recall);
    }else if(mode_ == 1) {
        report.charF = report.precision;
    }else if(mode_ == 2) {
        report.charF = report.recall;
    }

    report.ref_len = vals_[2];
    report.sys_len = vals_[1];

    return report;
}




EvalMeasureCharF::EvalMeasureCharF(const std::string & config, const cnn::Dict & vocab) : smooth_val_(0),mode_(0),ignore_bpe_(true),use_space_(false),ngram_order_(6), beta_(3),vocab_(vocab) {
    if(config.length() == 0) return;
    for(const EvalMeasure::StringPair & strs : EvalMeasure::ParseConfig(config)) {
        if(strs.first == "order") {
            ngram_order_ = boost::lexical_cast<int>(strs.second);
        } else if(strs.first == "beta") {
            beta_ = boost::lexical_cast<float>(strs.second);
        } else if(strs.first == "smooth") {
            smooth_val_ = boost::lexical_cast<float>(strs.second);
        } else if(strs.first == "use_space") {
            use_space_ = boost::lexical_cast<bool>(strs.second);
        } else if(strs.first == "ignore_bpe") {
            ignore_bpe_ = boost::lexical_cast<bool>(strs.second);
        } else if(strs.first == "mode") {
            mode_ = boost::lexical_cast<int>(strs.second);
        } else {
            THROW_ERROR("Bad configuration string: " << config);
        }
    }
}


