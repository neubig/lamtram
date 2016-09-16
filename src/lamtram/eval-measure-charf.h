#ifndef EVAL_MEASURE_CHARF_H__
#define EVAL_MEASURE_CHARF_H__

// This class calculate the CHARF evaulation measure as proposed by


#include <lamtram/eval-measure.h>
#include <map>
#include <vector>

namespace lamtram {

typedef struct {
    float charF; // CharF score
    float ref_len, sys_len; // Reference and system, lengths
    float precision,recall; // The ref/sys ratio and brevity penalty
} CharFReport;

class EvalStatsCharF : public EvalStats {
public:
    EvalStatsCharF(const std::vector<EvalStatsDataType> vals = std::vector<EvalStatsDataType>(),
                  float beta = 0.5)
            : beta_(beta) {
        vals_ = vals;
    }
    virtual std::string GetIdString() const { return "CHARF"; }
    virtual float ConvertToScore() const;
    virtual std::string ConvertToString() const;
    virtual EvalStatsPtr Clone() const { return EvalStatsPtr(new EvalStatsCharF(vals_, beta_)); }
    CharFReport CalcCharFReport() const;
/*    float GetAvgLogPrecision() const;
    float GetLengthRatio() const;
    int GetNgramOrder() const { return vals_.size()/3; }
    float GetMatch(int n) const { return vals_[3*n]+smooth_; }
    float GetSysCount(int n) const { return vals_[3*n+1]+smooth_; }
    float GetRefCount(int n) const { return vals_[3*n+2]+smooth_; } */
private:
    float beta_;
    //vals: 0 - #correct 1 - #hyp 2 - #ref
    
};



class EvalMeasureCharF : public EvalMeasure {

public:

    // NgramStats are a mapping between ngrams and the number of occurrences
    typedef std::map<std::vector<char>,int> NgramStats;

    // A cache to hold the stats
    typedef std::map<int,std::shared_ptr<NgramStats> > StatsCache;


    EvalMeasureCharF(int ngram_order = 4, float beta = 0.5,const cnn::Dict & vocab) : 
        ngram_order_(ngram_order), beta_(beta) ,vocab_(vocab){ }
    EvalMeasureCharF(const std::string & config,const cnn::Dict & vocab);

    // Calculate the stats for a single sentence
    virtual std::shared_ptr<EvalStats> CalculateStats(
                const Sentence & ref,
                const Sentence & sys,
                int ref_cache_id = INT_MAX,
                int sys_cache_id = INT_MAX);
    
    // Calculate the stats for a single sentence
    virtual std::shared_ptr<EvalStats> CalculateStats(
                const Sentence & ref,
                const Sentence & sys) const;

    // Calculate the stats with cached n-grams
    std::shared_ptr<EvalStats> CalculateStats(
                        const NgramStats & ref_ngrams,
                        int ref_len,
                        const NgramStats & sys_ngrams,
                        int sys_len) const; 


    // Calculate the stats for a single sentence
    virtual EvalStatsPtr ReadStats(
                const std::string & file);

    // Clear the ngram cache
    virtual void ClearCache() { cache_.clear(); }
    
    // Calculate the n-gram statistics necessary for BLEU in advance
    NgramStats * ExtractNgrams(const Sentence & sentence) const;
    
    

    int GetNgramOrder() const { return ngram_order_; }
    void SetNgramOrder(int ngram_order) { ngram_order_ = ngram_order; }
    float GetBetaVal() const { return beta_; }
    void BetaVal(float beta) { beta_ = beta; }
    std::string GetIdString() { return "CHARF"; }
protected:

    int ngram_order_;
    float beta_;

    const cnn::Dict & vocab_;

    // A cache to hold the stats
    StatsCache cache_;

    // Get the stats that are in a cache
    std::shared_ptr<NgramStats> GetCachedStats(const Sentence & sent, int cache_id);


};

}

#endif
