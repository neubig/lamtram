#ifndef EVAL_MEASURE_INTERP_H__
#define EVAL_MEASURE_INTERP_H__

// In interpreted version of several evaluation measures
// Specify it as follows
//  interp=0.4|bleu|0.6|ribes
// To do the 0.4/0.6 interperation fo the BLEU and RIBES measures

#include <lamtram/sentence.h>
#include <lamtram/eval-measure.h>
#include <dynet/dict.h>
#include <map>
#include <vector>

namespace lamtram {

// The interpolated stats
class EvalStatsInterp : public EvalStats {
public:
    EvalStatsInterp(const std::vector<EvalStatsPtr> & stats = std::vector<EvalStatsPtr>(),
                    const std::vector<float> & coeffs = std::vector<float>()) :
        stats_(stats), coeffs_(coeffs) { }
    virtual ~EvalStatsInterp() { }

    virtual std::string ConvertToString() const;
    virtual std::string GetIdString() const;
    virtual float ConvertToScore() const;
    // Check if the value is zero
    virtual bool IsZero();
    virtual EvalStats & PlusEquals(const EvalStats & rhs);
    virtual EvalStats & TimesEquals(EvalStatsDataType mult);
    virtual bool Equals(const EvalStats & rhs) const;

    EvalStatsPtr Clone() const;

    virtual std::string WriteStats();

protected:
    std::vector<EvalStatsPtr> stats_;
    std::vector<float> coeffs_;
};

// The interpolated evaluation measure
class EvalMeasureInterp : public EvalMeasure {

public:


    EvalMeasureInterp(const std::vector<std::shared_ptr<EvalMeasure> > & measures, const std::vector<float> & coeffs) 
        : measures_(measures), coeffs_(coeffs) { }
    EvalMeasureInterp(const std::string & str, const dynet::Dict & vocab);
    virtual ~EvalMeasureInterp() { }

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
    std::vector<std::shared_ptr<EvalMeasure> > measures_;
    std::vector<float> coeffs_;

};

}

#endif
