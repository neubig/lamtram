#include <lamtram/eval-measure-charf.h>
#include <boost/lexical_cast.hpp>
#include <lamtram/macros.h>
using namespace std;
using namespace lamtram;
using namespace boost;



EvalMeasureCharF::NgramStats * EvalMeasureCharF::ExtractNgrams(const Sentence & sentence) const {
    EvalMeasureCharF::NgramStats * all_ngrams = new EvalMeasureCharF::NgramStats;
    
    for(int i =0; i < sentence.size()-k; i++) {
        cout << vocab_.convert(sentence[i]) << " " << endl;
    }
    exit(-1);
    // TODO
/*    vector<WordId> ngram;
    for (int k = 0; k < ngram_order_; k++) {
        for(int i =0; i < max((int)sentence.size()-k,0); i++) {
            for ( int j = i; j<= i+k; j++) {
                ngram.push_back(sentence[j]);
            }
            ++((*all_ngrams)[ngram]);
            ngram.clear();
        }
    }*/
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
    vals[1] = sys_ngrams.size();
    vals[1] = ref_ngrams.size();
    
/*
    for (NgramStats::const_iterator it = sys_ngrams.begin(); it != sys_ngrams.end(); it++) {
        NgramStats::const_iterator ref_it = ref_ngrams.find(it->first);
        if(ref_it != ref_ngrams.end()) {
            vals[3* (it->first.size()-1)] += min(ref_it->second,it->second);
        }
    }
    */
    // Create the stats for this sentence
    EvalStatsPtr ret(new EvalStatsCharF(vals, beta_));
    // If we are using sentence based, take the average immediately
    return ret;
}


// Read in the stats
std::shared_ptr<EvalStats> EvalMeasureCharF::ReadStats(const std::string & line) {
    EvalStatsPtr ret;
    ret.reset(new EvalStatsCharF(std::vector<EvalStatsDataType>(), beta_));
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

    report.precision = 1.0 * vals_[0] / vals_[2];
    report.recall = 1.0 * vals_[0] / vals_[1];
    report.charF = 1.0 *(1+beta_*beta_) * (report.precision * report.recall) /(beta_*beta_ * report.precision + report.recall);


    return report;
}




EvalMeasureCharF::EvalMeasureCharF(const std::string & config, const cnn::Dict & vocab) : ngram_order_(3), beta_(0.5),vocab_(vocab) {
    if(config.length() == 0) return;
    for(const EvalMeasure::StringPair & strs : EvalMeasure::ParseConfig(config)) {
        if(strs.first == "order") {
            ngram_order_ = boost::lexical_cast<int>(strs.second);
        } else if(strs.first == "beta") {
            beta_ = boost::lexical_cast<float>(strs.second);
        } else {
            THROW_ERROR("Bad configuration string: " << config);
        }
    }
}
