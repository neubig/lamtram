#ifndef EVAL_MEASURE_H__
#define EVAL_MEASURE_H__

#include <lamtram/sentence.h>
#include <cfloat>
#include <climits>
#include <vector>
#include <string>
#include <sstream>
#include <memory>

namespace lamtram {

// An interface for holding stats for a particular evaluation measure
class EvalStats;
typedef float EvalStatsDataType;
typedef std::shared_ptr<EvalStats> EvalStatsPtr;
class EvalStats {
public:
    EvalStats() { }
    EvalStats(const std::vector<EvalStatsDataType> & vals) : vals_(vals) { }
    virtual ~EvalStats() { }
    // ConvertToScore takes the stats and converts them into a score
    virtual float ConvertToScore() const = 0;
    // Clone is a utility function that basically calls the constructor of
    // child classes, as this functionality is not included in C++
    virtual EvalStatsPtr Clone() const = 0;
    // Get the ID of this stats
    virtual std::string GetIdString() const = 0;
    // Check if the value is zero
    virtual bool IsZero();
    // Utility functions
    virtual std::string ConvertToString() const;
    virtual EvalStats & PlusEquals(const EvalStats & rhs);
    virtual EvalStats & PlusEqualsTimes(const EvalStats & rhs, float p);
    virtual EvalStats & TimesEquals(EvalStatsDataType mult);
    virtual EvalStatsPtr Plus(const EvalStats & rhs);
    virtual EvalStatsPtr Times(EvalStatsDataType mult);
    virtual bool Equals(const EvalStats & rhs) const;
    const std::vector<EvalStatsDataType> & GetVals() const;
    virtual void ReadStats(const std::string & str);
    virtual std::string WriteStats();
protected:
    std::vector<EvalStatsDataType> vals_;
};

inline bool operator==(const EvalStats & lhs, const EvalStats & rhs) {
    return lhs.Equals(rhs);
}
inline bool operator==(const EvalStatsPtr & lhs, const EvalStatsPtr & rhs) {
    return *lhs == *rhs;
}
inline std::ostream &operator<<( std::ostream &out, const EvalStats &L ) {
    out << L.ConvertToString();
    return out;
}

// Simple sentence-averaged stats
class EvalStatsAverage : public EvalStats {
public:
    EvalStatsAverage(float val = 0.0, float denom = 1.0) {
        vals_.resize(2);
        vals_[0] = val;
        vals_[1] = denom;
    }
    float ConvertToScore() const { return vals_[1] ? vals_[0]/vals_[1] : 0; }
    EvalStatsPtr Clone() const { return EvalStatsPtr(new EvalStatsAverage(vals_[0], vals_[1])); }
    virtual std::string GetIdString() const { return "AVG"; }
    // Getters
    float GetVal() const { return vals_[0]; }
    int GetDenom() const { return vals_[1]; }
private:
};

// An interface for an evaluation measure. All evaluation measures
// must implement CalculateStats(), which returns an EvalStats object
// for measuring the distance between two sentences
class EvalMeasure {

public:

    typedef std::pair<std::string, std::string> StringPair;

    // Create with default settings
    EvalMeasure() : factor_(0) { }
    // Create with configuration
    EvalMeasure(const std::string & config) : factor_(0) { }
    virtual ~EvalMeasure() { }

    // Calculate the stats for a single sentence
    virtual EvalStatsPtr CalculateStats(
                const Sentence & ref,
                const Sentence & sys) const = 0;

    // Calculate the stats for a single sentence
    virtual EvalStatsPtr CalculateCachedStats(
                const Sentence & ref,
                const Sentence & sys,
                int ref_cache_id = INT_MAX,
                int sys_cache_id = INT_MAX) {
        return CalculateStats(ref,sys);
    }

    // Calculate the stats for a singles sentence with factors
    virtual EvalStatsPtr CalculateCachedStats(
                const std::vector<Sentence> & refs,
                const std::vector<Sentence> & syss,
                int ref_cache_id = INT_MAX,
                int sys_cache_id = INT_MAX) {
        return CalculateCachedStats(refs[factor_],syss[factor_],ref_cache_id,sys_cache_id);
    }

    // Calculate the stats for a single sentence
    virtual EvalStatsPtr ReadStats(
                const std::string & file) = 0;

    // Parse a pair of strings
    static std::vector<StringPair> ParseConfig(const std::string & config);

    // Clear the cache
    virtual void ClearCache() { }

protected:

    // Which factore to calculate over
    int factor_;

};

}

#endif
