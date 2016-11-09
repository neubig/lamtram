#ifndef EVAL_MEASURE_FSCORE_H__
#define EVAL_MEASURE_FSCORE_H__

// This class calculate the FSCORE 


#include <lamtram/eval-measure.h>
#include <map>
#include <vector>
#include <dynet/dict.h>

namespace lamtram {

#define EOS 0

typedef struct {
    float fscore; // Fscore score
    float ref_len, sys_len; // Reference and system, lengths
    float precision,recall; // The ref/sys ratio and brevity penalty
} FscoreReport;

class EvalStatsFscore : public EvalStats {
public:
    EvalStatsFscore(const std::vector<EvalStatsDataType> vals = std::vector<EvalStatsDataType>(),
                                    float smooth = 0, float beta = 1,int mode = 0)
            : smooth_(smooth),beta_(beta),mode_(mode) {
        vals_ = vals;
    }
    virtual std::string GetIdString() const { return "FSCORE"; }
    virtual float ConvertToScore() const;
    virtual std::string ConvertToString() const;
    virtual EvalStatsPtr Clone() const { return EvalStatsPtr(new EvalStatsFscore(vals_, smooth_,beta_,mode_)); }
    FscoreReport CalcFscoreReport() const;
/*    float GetAvgLogPrecision() const;
    float GetLengthRatio() const;
    int GetNgramOrder() const { return vals_.size()/3; }
    float GetMatch(int n) const { return vals_[3*n]+smooth_; }
    float GetSysCount(int n) const { return vals_[3*n+1]+smooth_; }
    float GetRefCount(int n) const { return vals_[3*n+2]+smooth_; } */
private:
    float beta_;
    int mode_;
    //vals: 0 - #correct 1 - #hyp 2 - #ref
    float smooth_;
    
};



class EvalMeasureFscore : public EvalMeasure {

public:

    // NgramStats are a mapping between ngrams and the number of occurrences
    typedef std::map<std::string,int> NgramStats;

    // A cache to hold the stats
    typedef std::map<int,std::shared_ptr<NgramStats> > StatsCache;


    EvalMeasureFscore(const dynet::Dict & vocab, float smooth_val = 0, int ngram_order = 1, float beta = 1,int mode = 0) : 
        smooth_val_(smooth_val),ngram_order_(ngram_order), beta_(beta) ,vocab_(vocab),mode_(mode){ }
    EvalMeasureFscore(const std::string & config,const dynet::Dict & vocab);

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
    float GetSmoothVal() const { return smooth_val_; }
    void SetSmoothVal(float smooth_val) { smooth_val_ = smooth_val; }
    std::string GetIdString() { return "FSCORE"; }
protected:

    int ngram_order_;
    float beta_;

    //mode 0 - f-score with beta; 1 - precision; 2 - recall
    int mode_;

    // The amount by which to smooth the counts
    float smooth_val_;

    const dynet::Dict & vocab_;

    // A cache to hold the stats
    StatsCache cache_;

    // Get the stats that are in a cache
    std::shared_ptr<NgramStats> GetCachedStats(const Sentence & sent, int cache_id);


};

}

#endif
