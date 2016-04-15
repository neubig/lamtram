#include <lamtram/eval-measure.h>

#include <lamtram/macros.h>
#include <boost/algorithm/string.hpp>

#include <fstream>
#include <map>

using namespace std;
using namespace lamtram;
using namespace boost;

#define NGRAM_ORDER 5
#define NBEST_COUNT 10
#define POP_LIMIT 500

bool EvalStats::IsZero() {
    for(const EvalStatsDataType & val : vals_)
        if(val != 0)
            return false;
    return true;
}
// Utility functions
std::string EvalStats::ConvertToString() const {
    std::ostringstream oss;
    oss << GetIdString() << " = " << ConvertToScore();
    return oss.str();
}
EvalStats & EvalStats::PlusEquals(const EvalStats & rhs) {
    if(vals_.size() == 0) {
        vals_ = rhs.vals_;
    } else if (rhs.vals_.size() != 0) {
        if(rhs.vals_.size() != vals_.size())
            THROW_ERROR("Mismatched in EvalStats::PlusEquals");
        for(int i = 0; i < (int)rhs.vals_.size(); i++)
            vals_[i] += rhs.vals_[i];
    }
    return *this;
}
EvalStats & EvalStats::PlusEqualsTimes(const EvalStats & rhs, float p) {
    if(vals_.size() == 0) {
        vals_ = rhs.vals_;
        for(int i = 0; i < (int)vals_.size(); i++)
            vals_[i] *= p;
    } else if (rhs.vals_.size() != 0) {
        if(rhs.vals_.size() != vals_.size())
            THROW_ERROR("Mismatched in EvalStats::PlusEqualsTimes");
        for(int i = 0; i < (int)rhs.vals_.size(); i++)
            vals_[i] += rhs.vals_[i] * p;
    }
    return *this;
}
EvalStats & EvalStats::TimesEquals(EvalStatsDataType mult) {
    for(EvalStatsDataType & val : vals_)
        val *= mult;
    return *this;
}
EvalStatsPtr EvalStats::Plus(const EvalStats & rhs) {
    EvalStatsPtr ret(this->Clone());
    ret->PlusEquals(rhs);
    return ret;
}
EvalStatsPtr EvalStats::Times(EvalStatsDataType mult) {
    EvalStatsPtr ret(this->Clone());
    ret->TimesEquals(mult);
    return ret;
}
bool EvalStats::Equals(const EvalStats & rhs) const {
    if(vals_.size() != rhs.vals_.size()) return false;
    for(int i = 0; i < (int)vals_.size(); i++) {
        if(fabs(vals_[i]-rhs.vals_[i]) > 1e-6)
            return false;
    }
    return true;
}
const std::vector<EvalStatsDataType> & EvalStats::GetVals() const { return vals_; }
void EvalStats::ReadStats(const std::string & str) {
    vals_.resize(0);
    EvalStatsDataType val;
    std::istringstream iss(str);
    while(iss >> val)
        vals_.push_back(val);
}
std::string EvalStats::WriteStats() {
    std::ostringstream oss;
    for(int i = 0; i < (int)vals_.size(); i++) {
        if(i) oss << ' ';
        oss << vals_[i];
    }
    return oss.str();
}

vector<EvalMeasure::StringPair> EvalMeasure::ParseConfig(const string & str) {
    vector<string> arr1, arr2;
    boost::split ( arr1, str, boost::is_any_of(","));
    vector<EvalMeasure::StringPair> ret;
    for(const std::string & my_str : arr1) {
        boost::split ( arr2, my_str, boost::is_any_of("="));
        if(arr2.size() != 2)
            THROW_ERROR("Bad evaluation measure config:" << str);
        ret.push_back(make_pair(arr2[0], arr2[1]));
    }
    return ret;
}
