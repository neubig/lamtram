#ifndef EVAL_MEASURE_LOADER_H__
#define EVAL_MEASURE_LOADER_H__

#include <dynet/dict.h>
#include <string>

namespace lamtram {

class EvalMeasure;

class EvalMeasureLoader {

    EvalMeasureLoader(); // = delete;
    EvalMeasureLoader(const EvalMeasureLoader &); // = delete;
    EvalMeasureLoader & operator=(const EvalMeasureLoader &); // = delete;

public:
    // Create measure from string
    static EvalMeasure * CreateMeasureFromString(const std::string & str, const dynet::Dict & vocab);

}; // class EvalMeasureLoader

} // namespace lamtram

#endif // EVAL_MEASURE_LOADER_H__

