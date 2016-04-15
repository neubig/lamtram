
#include <lamtram/eval-measure-ribes.h>
#include <lamtram/macros.h>
#include <boost/lexical_cast.hpp>
#include <cmath>

using namespace std;
using namespace lamtram;
using namespace boost;

// Measure the score of the sys output according to the ref
std::shared_ptr<EvalStats> EvalMeasureRibes::CalculateStats(const Sentence & ref, const Sentence & sys) const {

    // check reference length, if zero, return 1 only if system is also empty
    if(ref.size() == 0)
        return std::shared_ptr<EvalStats>(new EvalStatsRibes((sys.size() == 0 ? 1 : 0), 1));

    // check hypothesis length, return "zeros" if no words are found
    if(sys.size() == 0)
        return std::shared_ptr<EvalStats>(new EvalStatsRibes(0, 1));
    
    // calculate brevity penalty (BP), not exceeding 1.0
    float bp = min(1.0, exp(1.0 - 1.0 * ref.size()/sys.size())); 
    
    // determine which ref. word corresponds to each sysothesis word
    // list for ref. word indices
    vector<int> intlist;
    
    // Find the positions of each word in each of the sentences
    std::unordered_map<WordId, vector<int> > ref_count, sys_count;
    for(int i = 0; i < (int)ref.size(); i++)
        ref_count[ref[i]].push_back(i);
    for(int i = 0; i < (int)sys.size(); i++)
        sys_count[sys[i]].push_back(i);
    
    for(int i = 0; i < (int)sys.size(); i++) {
        // If sys[i] doesn't exist in the reference, go to the next word
        if(ref_count.find(sys[i]) == ref_count.end())
            continue;
        // Get matched words
        const vector<int> & ref_match = ref_count[sys[i]];
        const vector<int> & sys_match = sys_count[sys[i]];

        // if we can determine one-to-one word correspondence by only unigram
        // one-to-one correspondence
        if (ref_match.size() == 1 && sys_match.size() == 1) {
            intlist.push_back(ref_match[0]);
        // if not, we consider context words
        } else {
            // These vectors store all hypotheses that are still matching on the right or left
            vector<int> left_ref = ref_match, left_sys = sys_match,
                        right_ref = ref_match, right_sys = sys_match;
            for(int window = 1; window < max(i, (int)sys.size()-i); window++) {
                // Update the possible hypotheses on the left
                if(window <= i) {
                    vector<int> new_left_ref, new_left_sys;
                    for(int j : left_ref)
                        if(window <= j && ref[j-window] == sys[i-window])
                            new_left_ref.push_back(j);
                    for(int j : left_sys)
                        if(window <= j && sys[j-window] == sys[i-window])
                            new_left_sys.push_back(j);
                    if(new_left_ref.size() == 1 && new_left_sys.size() == 1) {
                        intlist.push_back(new_left_ref[0]);
                        break;
                    }
                    left_ref = new_left_ref; left_sys = new_left_sys;
                }
                // Update the possible hypotheses on the right
                if(i+window < (int)sys.size()) {
                    vector<int> new_right_ref, new_right_sys;
                    for(int j : right_ref)
                        if(j+window < (int)ref.size() && ref[j+window] == sys[i+window])
                            new_right_ref.push_back(j);
                    for(int j : right_sys)
                        if(j+window < (int)sys.size() && sys[j+window] == sys[i+window])
                            new_right_sys.push_back(j);
                    if(new_right_ref.size() == 1 && new_right_sys.size() == 1) {
                        intlist.push_back(new_right_ref[0]);
                        break;
                    }
                    right_ref = new_right_ref; right_sys = new_right_sys;
                }
            }
        }
    }
    // cerr << "intlist:"; for(int i : intlist) cerr << " " << i; cerr << endl;
    
    // At least two word correspondences are needed for rank correlation
    int n = intlist.size();
    if (n == 1 && ref.size() == 1)
        return std::shared_ptr<EvalStats>(new EvalStatsRibes(1.0 * (pow(1.0/sys.size(), alpha_)) * (pow(bp, beta_)), 1));
    // if not, return score 0.0
    else if(n < 2)
        return std::shared_ptr<EvalStats>(new EvalStatsRibes(0, 1));
    
    // calculation of rank correlation coefficient
    // count "ascending pairs" (intlist[i] < intlist[j])
    int ascending = 0;
    for(int i = 0; i < (int)intlist.size()-1; i++)
        for(int j = i+1; j < (int)intlist.size(); j++)
            if(intlist[i] < intlist[j])
                ascending++;
    
    // normalize Kendall's tau
    float nkt = float(ascending) / ((n * (n - 1))/2);
    
    // calculate unigram precision
    float precision = 1.0 * n / sys.size();
    
    // RIBES = (normalized Kendall's tau) * (unigram_precision ** alpha) * (brevity_penalty ** beta)
    return std::shared_ptr<EvalStats>(new EvalStatsRibes(nkt * (pow(precision, alpha_)) * (pow(bp, beta_)), 1));

}

// Read in the stats
std::shared_ptr<EvalStats> EvalMeasureRibes::ReadStats(const std::string & line) {
    EvalStatsPtr ret(new EvalStatsRibes());
    ret->ReadStats(line);
    return ret;
}



EvalMeasureRibes::EvalMeasureRibes(const std::string & config)
                        : RIBES_VERSION_("1.02.3"), alpha_(0.25), beta_(0.1) {
    if(config.length() == 0) return;
    for(const EvalMeasure::StringPair & strs : EvalMeasure::ParseConfig(config)) {
        if(strs.first == "alpha") {
            alpha_ = boost::lexical_cast<float>(strs.second);
        } else if(strs.first == "beta") {
            beta_ = boost::lexical_cast<float>(strs.second);
        } else if(strs.first == "factor") {
            factor_ = boost::lexical_cast<int>(strs.second);
        } else {
            THROW_ERROR("Bad configuration string: " << config);
        }
    }
}
