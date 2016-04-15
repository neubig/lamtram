
#include <lamtram/eval-measure-wer.h>
#include <lamtram/macros.h>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace lamtram;
using namespace boost;

// Calculate Levenshtein Distance
int EvalMeasureWer::EditDistance(const Sentence & ref, const Sentence & sys) const {
    int rs = ref.size()+1, ss = sys.size()+1;
    vector<int> dists(rs*ss);
    for(int i = 0; i < rs; i++)
        dists[i*ss] = i;
    for(int j = 1; j < ss; j++)
        dists[j] = j;
    for(int i = 1; i < rs; i++)
        for(int j = 1; j < ss; j++) {
            dists[i*ss+j] =
                min(
                    min(dists[(i-1)*ss+j]+1, dists[i*ss+j-1]+1),
                    dists[(i-1)*ss+j-1] + (ref[i-1] == sys[j-1] ? 0 : 1));
        }
    return dists[rs*ss-1];
}

// Measure the score of the sys output according to the ref
std::shared_ptr<EvalStats> EvalMeasureWer::CalculateStats(const Sentence & ref, const Sentence & sys) const {

    return std::shared_ptr<EvalStats>(new EvalStatsWer(EditDistance(ref,sys), ref.size(), reverse_));

}

// Read in the stats
std::shared_ptr<EvalStats> EvalMeasureWer::ReadStats(const std::string & line) {
    EvalStatsPtr ret(new EvalStatsWer(0, 0, reverse_));
    ret->ReadStats(line);
    return ret;
}


EvalMeasureWer::EvalMeasureWer(const std::string & config)
                        : reverse_(false) {
    if(config.length() == 0) return;
    for(const EvalMeasure::StringPair & strs : EvalMeasure::ParseConfig(config)) {
        if(strs.first == "reverse") {
            if(strs.second == "true")
                reverse_ = true;
            else if(strs.second == "false")
                reverse_ = false;
            else
                THROW_ERROR("Bad reverse value: " << strs.second);
        } else if(strs.first == "factor") {
            factor_ = boost::lexical_cast<int>(strs.second);
        } else {
            THROW_ERROR("Bad configuration string: " << config);
        }
    }
}
