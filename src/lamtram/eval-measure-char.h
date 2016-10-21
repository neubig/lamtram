#ifndef EVAL_MEASURE_CHAR_H__
#define EVAL_MEASURE_CHAR_H__

// Port word based metrics to char-based metrics
// Specify it as follows
//  char=bleu,
//

#include <lamtram/sentence.h>
#include <lamtram/eval-measure.h>
#include <dynet/dict.h>
#include <map>
#include <vector>
#include <string-util.h>

namespace lamtram {

#define EOS 0


// The interpolated evaluation measure
class EvalMeasureChar : public EvalMeasure {

public:


    EvalMeasureChar(const std::shared_ptr<EvalMeasure> measure,const dynet::Dict & vocab) 
        : measure_(measure),vocab_(vocab) { }
    EvalMeasureChar(const std::string & str, const dynet::Dict & vocab);
    virtual ~EvalMeasureChar() { }

    // Calculate the stats for a single sentence
    virtual std::shared_ptr<EvalStats> CalculateStats(
                const Sentence & ref,
                const Sentence & sys) const;
    virtual EvalStatsPtr CalculateCachedStats(
                const std::vector<Sentence> & ref,
                const std::vector<Sentence> & syss,
                int ref_cache_id = INT_MAX,
                int sys_cache_id = INT_MAX);

    // Calculate the stats for a single sentence
    virtual EvalStatsPtr ReadStats(
                const std::string & file);

protected:
    std::shared_ptr<EvalMeasure> measure_;

    std::map<std::string,int> voc_;
    bool use_space_;
    bool ignore_bpe_;
    const dynet::Dict & vocab_;


    Sentence ConvertSentence(const Sentence & sent,std::map<std::string,int> & dict) const;
};

}

#endif
