#include <lamtram/eval-measure-loader.h>

#include <lamtram/eval-measure.h>
#include <lamtram/eval-measure-bleu.h>
#include <lamtram/eval-measure-extern.h>
#include <lamtram/eval-measure-ribes.h>
#include <lamtram/eval-measure-wer.h>
#include <lamtram/eval-measure-interp.h>
#include <lamtram/macros.h>
#include <dynet/dict.h>

using namespace std;
using namespace lamtram;

namespace lamtram {

EvalMeasure * EvalMeasureLoader::CreateMeasureFromString(const string & str, const dynet::Dict & vocab) {
    // Get the eval, config substr
    string eval, config;
    size_t eq = str.find(':');
    if(eq == string::npos) { eval = str; }
    else { eval = str.substr(0,eq); config = str.substr(eq+1); }
    // Create the actual measure
    if(eval == "bleu") 
        return new EvalMeasureBleu(config);
    else if(eval == "extern")
        return new EvalMeasureExtern(config, vocab);
    else if(eval == "ribes")
        return new EvalMeasureRibes(config);
    else if(eval == "wer")
        return new EvalMeasureWer(config);
    else if(eval == "interp")
        return new EvalMeasureInterp(config, vocab);
    else
        THROW_ERROR("Unknown evaluation measure: " << eval);
    return NULL;
}

} // namespace lamtram

