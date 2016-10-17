#include <lamtram/eval-measure-extern.h>
#include <lamtram/macros.h>
#include <dynet/dict.h>
#include <boost/algorithm/string.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/lexical_cast.hpp>

// External scoring code is adapted from Moses, which is also LGPL 2.1:
// github.com/moses-smt/mosesdecoder/blob/master/mert/MeteorScorer.cpp

using namespace std;
using namespace lamtram;
using namespace boost;
using namespace boost::iostreams;

#define CHILD_STDIN_READ pipefds_input[0]
#define CHILD_STDIN_WRITE pipefds_input[1]
#define CHILD_STDOUT_READ pipefds_output[0]
#define CHILD_STDOUT_WRITE pipefds_output[1]

EvalMeasureExtern::EvalMeasureExtern(const std::string & config, const dynet::Dict & vocab)
                        : vocab_(vocab), run_(""), eos_(false) {
    if(config.length() == 0) THROW_ERROR("Required for external measure: run");
    for(const EvalMeasure::StringPair & strs : EvalMeasure::ParseConfig(config)) {
        if(strs.first == "run") {
            run_ = strs.second;
        } else if(strs.first == "eos") {
            if(strs.second == "true")
                eos_ = true;
            else if(strs.second == "false")
                eos_ = false;
            else
                THROW_ERROR("Bad eos value: " << strs.second);
        } else {
            THROW_ERROR("Bad configuration string: " << config);
        }
    }
    if(run_ == "") THROW_ERROR("Required for external measure: run");

    // Create pipes for process communication
    int pipe_status;
    int pipefds_input[2];
    int pipefds_output[2];
    pipe_status = pipe(pipefds_input);
    if (pipe_status == -1) {
        THROW_ERROR("Error creating pipe");
    }
    pipe_status = pipe(pipefds_output);
    if (pipe_status == -1) {
        THROW_ERROR("Error creating pipe");
    }
    // Fork
    pid_t pid;
    pid = fork();
    if (pid == pid_t(0)) {
        // Child's IO
        dup2(CHILD_STDIN_READ, 0);
        dup2(CHILD_STDOUT_WRITE, 1);
        close(CHILD_STDIN_WRITE);
        close(CHILD_STDOUT_READ);
        // Execute external command: the format is executable followed by args
        // followed by null.  In this case, the only arg is the executable
        // itself (conventionally passed as arg0)
        execl(run_.c_str(), run_.c_str(), (char*) NULL);
        THROW_ERROR("Continued after execl");
    }
    // Parent's IO
    close(CHILD_STDIN_READ);
    close(CHILD_STDOUT_WRITE);
    // IO streams for process communication
    to_child_buffer_.reset(new stream_buffer<file_descriptor_sink>(CHILD_STDIN_WRITE, file_descriptor_flags::close_handle));
    from_child_buffer_.reset(new stream_buffer<file_descriptor_source>(CHILD_STDOUT_READ, file_descriptor_flags::close_handle));
    to_child_.reset(new ostream(to_child_buffer_.get()));
    from_child_.reset(new istream(from_child_buffer_.get()));
}

// Measure the score of the sys output according to the ref
std::shared_ptr<EvalStats> EvalMeasureExtern::CalculateStats(const Sentence & ref, const Sentence & sys) const {
    int offset = eos_ ? 0 : 1;
    vector<string> sys_words;
    for (int i = 0; i < sys.size() - offset; ++i)
        sys_words.push_back(vocab_.convert(sys[i]));
    vector<string> ref_words;
    for (int i = 0; i < ref.size() - offset; ++i)
        ref_words.push_back(vocab_.convert(ref[i]));
    //cerr << "TO ||| " << boost::algorithm::join(sys_words, " ") << " ||| " << boost::algorithm::join(ref_words, " ") << endl;
    *to_child_ << boost::algorithm::join(sys_words, " ") << " ||| " << boost::algorithm::join(ref_words, " ") << endl;
    string from_line;
    getline(*from_child_, from_line);
    //cerr << "FROM ||| " << from_line << endl;
    EvalStatsDataType score = lexical_cast<float>(from_line);
    return std::shared_ptr<EvalStats>(new EvalStatsExtern(score));
}

// Read in the stats
std::shared_ptr<EvalStats> EvalMeasureExtern::ReadStats(const std::string & line) {
    EvalStatsPtr ret(new EvalStatsExtern(0));
    ret->ReadStats(line);
    return ret;
}
