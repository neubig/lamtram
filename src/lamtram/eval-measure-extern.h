#ifndef EVAL_MEASURE_EXTERN_H__
#define EVAL_MEASURE_EXTERN_H__

#include <lamtram/sentence.h>
#include <lamtram/eval-measure.h>
#include <dynet/dict.h>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>
#include <vector>

namespace lamtram {

class EvalStatsExtern : public EvalStats {

public:

    EvalStatsExtern(float score = 0.0) {
        vals_.resize(1);
        vals_[0] = score;
    }
    float ConvertToScore() const { return vals_[0]; }
    EvalStatsPtr Clone() const { return EvalStatsPtr(new EvalStatsExtern(vals_[0])); }
    virtual std::string GetIdString() const { return "Extern"; }

};

class EvalMeasureExtern : public EvalMeasure {

public:

    EvalMeasureExtern(const std::string & str, const dynet::Dict & vocab);

    // Calculate the stats for a single sentence
    virtual std::shared_ptr<EvalStats> CalculateStats(
                const Sentence & ref,
                const Sentence & sys) const;

    // Calculate the stats for a single sentence
    virtual EvalStatsPtr ReadStats(
                const std::string & file);

protected:

    // Target vocabulary to generate sys/ref strings
    const dynet::Dict & vocab_;

    // External evaluation measure executable to run
    std::string run_;

    // Print <s> at end of sentence?  Defaults to false since most external
    // measures aren't expecting it.
    bool eos_;

    // External measure child process communication
    std::shared_ptr<boost::iostreams::stream_buffer<boost::iostreams::file_descriptor_sink>> to_child_buffer_;
    std::shared_ptr<boost::iostreams::stream_buffer<boost::iostreams::file_descriptor_source>> from_child_buffer_;
    std::shared_ptr<std::ostream> to_child_;
    std::shared_ptr<std::istream> from_child_;

};

}

#endif
