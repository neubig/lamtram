#pragma once

#include <lamtram/sentence.h>
#include <cnn/tensor.h>
#include <boost/program_options.hpp>
#include <string>

namespace cnn {
struct Trainer;
class Model;
class Dict;
}

namespace lamtram {

class EvalMeasure;


class LamtramTrain {

public:
    LamtramTrain() { }
    int main(int argc, char** argv);
    
    void TrainLM();
    void TrainEncDec();
    void TrainEncAtt();
    void TrainEncCls();

    // Bilingual maximum likelihood training
    template<class ModelType, class OutputType>
    void BilingualTraining(const std::vector<Sentence> & train_src,
                           const std::vector<OutputType> & train_trg,
                           const std::vector<OutputType> & train_cache,
                           const std::vector<Sentence> & dev_src,
                           const std::vector<OutputType> & dev_trg,
                           const cnn::Dict & vocab_src,
                           const cnn::Dict & vocab_trg,
                           cnn::Model & mod,
                           ModelType & encdec);

    // Minimum risk training
    template<class ModelType>
    void MinRiskTraining(const std::vector<Sentence> & train_src,
                         const std::vector<Sentence> & train_trg,
                         const std::vector<int> & train_fold_ids,                         
                         const std::vector<Sentence> & dev_src,
                         const std::vector<Sentence> & dev_trg,
                         const cnn::Dict & vocab_src,
                         const cnn::Dict & vocab_trg,
                         const EvalMeasure & eval,
                         cnn::Model & model,
                         ModelType & encdec);

    // Get the trainer to use
    typedef std::shared_ptr<cnn::Trainer> TrainerPtr;
    TrainerPtr GetTrainer(const std::string & trainer_id, const cnn::real learning_rate, cnn::Model & model);

    // Load in the training data
    void LoadFile(const std::string filename, bool add_last, cnn::Dict & vocab, std::vector<Sentence> & sents);
    void LoadLabels(const std::string filename, cnn::Dict & vocab, std::vector<int> & labs);

    void LoadBothFiles(
          const std::string filename_src, cnn::Dict & vocab_src, std::vector<Sentence> & sents_src,
          const std::string filename_trg, cnn::Dict & vocab_trg, std::vector<Sentence> & sents_trg);

protected:

    boost::program_options::variables_map vm_;

    // Variable settings
    cnn::real rate_thresh_, rate_decay_;
    int epochs_, context_, eval_every_;
    float scheduled_samp_, dropout_;
    std::string model_in_file_, model_out_file_;
    std::vector<std::string> train_files_trg_, train_files_src_;
    std::string dev_file_trg_, dev_file_src_;
    std::string softmax_sig_;

    std::vector<std::string> wildcards_;

};

}
