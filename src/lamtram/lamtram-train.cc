
#include <lamtram/lamtram-train.h>
#include <lamtram/neural-lm.h>
#include <lamtram/encoder-decoder.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/encoder-classifier.h>
#include <lamtram/macros.h>
#include <lamtram/timer.h>
#include <lamtram/model-utils.h>
#include <cnn/cnn.h>
#include <cnn/dict.h>
#include <cnn/random.h>
#include <cnn/training.h>
#include <cnn/tensor.h>
#include <boost/program_options.hpp>
#include <boost/range/irange.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace lamtram;
namespace po = boost::program_options;

int LamtramTrain::main(int argc, char** argv) {
  po::options_description desc("*** lamtram-train (by Graham Neubig) ***");
  desc.add_options()
    ("help", "Produce help message")
    ("train_trg", po::value<string>()->default_value(""), "Training file")
    ("dev_trg", po::value<string>()->default_value(""), "Development file")
    ("train_src", po::value<string>()->default_value(""), "Training source file for TMs")
    ("dev_src", po::value<string>()->default_value(""), "Development source file for TMs")
    ("eval_every", po::value<int>()->default_value(-1), "Evaluate every n sentences (-1 for full training set)")
    ("model_out", po::value<string>()->default_value(""), "File to write the model to")
    ("model_in", po::value<string>()->default_value(""), "If resuming training, read the model in")
    ("model_type", po::value<string>()->default_value("nlm"), "Model type (Neural LM nlm, Encoder Decoder encdec, Attentional Model encatt, or Encoder Classifier enccls)")
    ("epochs", po::value<int>()->default_value(100), "Number of epochs")
    ("rate_thresh",  po::value<double>()->default_value(1e-5), "Threshold for the learning rate")
    ("trainer", po::value<string>()->default_value("sgd"), "Training algorithm (sgd/momentum/adagrad/adadelta)")
    ("softmax", po::value<string>()->default_value("full"), "The type of softmax to use (full/hier/mod)")
    ("seed", po::value<int>()->default_value(0), "Random seed (default 0 -> changes every time)")
    ("learning_rate", po::value<double>()->default_value(0.1), "Learning rate")
    ("encoder_types", po::value<string>()->default_value("for"), "The type of encoder, multiple separated by a pipe (for=forward, rev=reverse)")
    ("context", po::value<int>()->default_value(2), "Amount of context information to use")
    ("wordrep", po::value<int>()->default_value(100), "Size of the word representations")
    ("layers", po::value<string>()->default_value("lstm:100:1"), "Descriptor for hidden layers, type:num_units:num_layers")
    ("cls_layers", po::value<string>()->default_value(""), "Descriptor for classifier layers, nodes1:nodes2:...")
    ("attention_nodes", po::value<int>()->default_value(100), "Number of nodes in the attention layer for encatt")
    ("cnn_mem", po::value<int>()->default_value(512), "How much memory to allocate to cnn")
    ("verbose", po::value<int>()->default_value(0), "How much verbose output to print")
    ;
  po::store(po::parse_command_line(argc, argv, desc), vm_);
  po::notify(vm_);   
  if (vm_.count("help")) {
    cout << desc << endl;
    return 1;
  }
  for(int i = 0; i < argc; i++) { cerr << argv[i] << " "; } cerr << endl;

  GlobalVars::verbose = vm_["verbose"].as<int>();

  // Set random seed if necessary
  int seed = vm_["seed"].as<int>();
  if(seed != 0) {
    delete cnn::rndeng;
    cnn::rndeng = new mt19937(seed);
  }

  // Sanity check for model type
  string model_type = vm_["model_type"].as<std::string>();
  if(model_type != "nlm" && model_type != "encdec" && model_type != "encatt" && model_type != "enccls") {
    cerr << desc << endl;
    THROW_ERROR("Model type must be neural LM (nlm) encoder decoder (encdec), attentional model (encatt), or encoder classifier (enccls)");
  }
  bool use_src = model_type == "encdec" || model_type == "enccls" || model_type == "encatt";

  // Other sanity checks
  try { train_file_trg_ = vm_["train_trg"].as<string>(); } catch(std::exception & e) { }
  try { dev_file_trg_ = vm_["dev_trg"].as<string>(); } catch(std::exception & e) { }
  try { model_out_file_ = vm_["model_out"].as<string>(); } catch(std::exception & e) { }
  if(!train_file_trg_.size())
    THROW_ERROR("Must specify a training file with --train_trg");
  if(!model_out_file_.size())
    THROW_ERROR("Must specify a model output file with --model_out");

  // Sanity checks for the source
  try { train_file_src_ = vm_["train_src"].as<string>(); } catch(std::exception & e) { }
  try { dev_file_src_ = vm_["dev_src"].as<string>(); } catch(std::exception & e) { }
  if(use_src && ((!train_file_src_.size()) || (dev_file_trg_.size() && !dev_file_src_.size())))
    THROW_ERROR("The specified model requires a source file to train, specify source files using train_src.");

  // Save some variables
  rate_thresh_ = vm_["rate_thresh"].as<double>();
  epochs_ = vm_["epochs"].as<int>();
  context_ = vm_["context"].as<int>();
  model_in_file_ = vm_["model_in"].as<string>();
  model_out_file_ = vm_["model_out"].as<string>();
  eval_every_ = vm_["eval_every"].as<int>();
  softmax_sig_ = vm_["softmax"].as<string>();

  // Perform appropriate training
  if(model_type == "nlm")       TrainLM();
  else if(model_type == "encdec")   TrainEncDec();
  else if(model_type == "encatt")   TrainEncAtt();
  else if(model_type == "enccls")   TrainEncCls();
  else                THROW_ERROR("Bad model type " << model_type);

  return 0;
}

void LamtramTrain::TrainLM() {

  // cnn::Dict
  DictPtr vocab_trg, vocab_src;
  std::shared_ptr<cnn::Model> model;
  std::shared_ptr<NeuralLM> nlm;
  if(model_in_file_.size()) {
    nlm.reset(ModelUtils::LoadMonolingualModel<NeuralLM>(model_in_file_, model, vocab_trg));
  } else {
    vocab_trg.reset(CreateNewDict());
    model.reset(new cnn::Model);
  }
  // if(!trg_sent) vocab_trg = cnn::Dict("");

  // Read the training files
  vector<Sentence> train_trg, dev_trg;
  LoadFile(train_file_trg_, true, *vocab_trg, train_trg); vocab_trg->Freeze(); vocab_trg->SetUnk("<unk>");
  if(dev_file_trg_.size()) LoadFile(dev_file_trg_, true, *vocab_trg, dev_trg);
  if(eval_every_ == -1) eval_every_ = train_trg.size();

  // Create the model
  if(model_in_file_.size() == 0)
    nlm.reset(new NeuralLM(vocab_trg, context_, 0, vm_["wordrep"].as<int>(), vm_["layers"].as<string>(), vocab_trg->GetUnkId(), softmax_sig_, *model));
  TrainerPtr trainer = GetTrainer(vm_["trainer"].as<string>(), vm_["learning_rate"].as<double>(), *model);
  
  // TODO: Learning rate
  cnn::real learning_rate = vm_["learning_rate"].as<double>();
  cnn::real learning_scale = 1.0;

  // Create a sentence list and random generator
  std::vector<int> train_ids(train_trg.size());
  std::iota(train_ids.begin(), train_ids.end(), 0);
  // Perform the training
  std::vector<cnn::expr::Expression> empty_hist;
  cnn::real last_ll = -1e99, best_ll = -1e99;
  bool do_dev = dev_trg.size() != 0;
  int loc = 0;
  int epoch = 0;
  std::shuffle(train_ids.begin(), train_ids.end(), *cnn::rndeng);
  while(epoch < epochs_) {
    // Start the training
    LLStats train_ll(nlm->GetVocabSize()), dev_ll(nlm->GetVocabSize());
    Timer time;
    for(auto id_pos : boost::irange(0, eval_every_)) {
      if(loc == (int)train_ids.size()) {
        // Shuffle the access order
        std::shuffle(train_ids.begin(), train_ids.end(), *cnn::rndeng);
        loc = 0;
        epoch++;
      }
      cnn::ComputationGraph cg;
      nlm->NewGraph(cg);
      nlm->BuildSentGraph(train_trg[train_ids[loc]], NULL, empty_hist, true, cg, train_ll);
      // cg.PrintGraphviz();
      train_ll.lik_ -= as_scalar(cg.forward());
      cg.backward();
      trainer->update();
      ++loc;
      if(loc % 100 == 0 || id_pos == eval_every_-1 || epochs_ == epoch) {
        double elapsed = time.Elapsed();
        cerr << "Epoch " << epoch+1 << " sent " << loc+1 << ": ppl=" << train_ll.CalcPPL() << ", unk=" << train_ll.unk_ << ", rate=" << learning_scale*learning_rate << ", time=" << elapsed << " (" << train_ll.words_/elapsed << " w/s)" << endl;
        if(epochs_ == epoch) break;
      }
    }
    // Measure development perplexity
    if(do_dev) {
      time = Timer();
      for(auto & sent : dev_trg) {
        cnn::ComputationGraph cg;
        nlm->NewGraph(cg);
        nlm->BuildSentGraph(sent, NULL, empty_hist, false, cg, dev_ll);
        dev_ll.lik_ -= as_scalar(cg.forward());
      }
      double elapsed = time.Elapsed();
      cerr << "Epoch " << epoch+1 << " dev: ppl=" << dev_ll.CalcPPL() << ", unk=" << dev_ll.unk_ << ", rate=" << learning_scale*learning_rate << ", time=" << elapsed << " (" << dev_ll.words_/elapsed << " w/s)" << endl;
    }
    // Adjust the learning rate
    trainer->update_epoch();
    // trainer->status(); cerr << endl;
    // Check the learning rate
    if(last_ll != last_ll)
      THROW_ERROR("Likelihood is not a number, dying...");
    cnn::real my_ll = do_dev ? dev_ll.lik_ : train_ll.lik_;
    if(my_ll < last_ll) {
      learning_scale /= 2.0;
    }
    last_ll = my_ll;
    if(best_ll < my_ll) {
      // Open the output stream
      ofstream out(model_out_file_.c_str());
      if(!out) THROW_ERROR("Could not open output file: " << model_out_file_);
      // Write the model (TODO: move this to a separate file?)
      WriteDict(*vocab_trg, out);
      // vocab_trg->Write(out);
      nlm->Write(out);
      ModelUtils::WriteModelText(out, *model);
      best_ll = my_ll;
    }
    // If the rate is less than the threshold
    if(learning_scale*learning_rate < rate_thresh_)
      break;
  }
}

void LamtramTrain::TrainEncDec() {

  // cnn::Dict
  DictPtr vocab_trg, vocab_src;
  std::shared_ptr<cnn::Model> model;
  std::shared_ptr<EncoderDecoder> encdec;
  if(model_in_file_.size()) {
    encdec.reset(ModelUtils::LoadBilingualModel<EncoderDecoder>(model_in_file_, model, vocab_src, vocab_trg));
  } else {
    vocab_src.reset(CreateNewDict());
    vocab_trg.reset(CreateNewDict());
    model.reset(new cnn::Model);
  }
  // if(!trg_sent) vocab_trg = cnn::Dict("");

  // Read the training files
  vector<Sentence> train_trg, dev_trg, train_src, dev_src;
  LoadFile(train_file_trg_, true, *vocab_trg, train_trg); vocab_trg->Freeze(); vocab_trg->SetUnk("<unk>");
  if(dev_file_trg_.size()) LoadFile(dev_file_trg_, true, *vocab_trg, dev_trg);
  LoadFile(train_file_src_, false, *vocab_src, train_src); vocab_src->Freeze(); vocab_src->SetUnk("<unk>");
  if(dev_file_src_.size()) LoadFile(dev_file_src_, false, *vocab_src, dev_src);
  if(eval_every_ == -1) eval_every_ = train_trg.size();

  // Create the model
  if(model_in_file_.size() == 0) {
    vector<LinearEncoderPtr> encoders;
    vector<string> encoder_types;
    boost::algorithm::split(encoder_types, vm_["encoder_types"].as<string>(), boost::is_any_of("|"));
    for(auto & spec : encoder_types) {
      LinearEncoderPtr enc(new LinearEncoder(vocab_src->size(), vm_["wordrep"].as<int>(), vm_["layers"].as<string>(), vocab_src->GetUnkId(), *model));
      if(spec == "for") { }
      else if(spec == "rev") { enc->SetReverse(true); }
      else { THROW_ERROR("Illegal encoder type: " << spec); }
      encoders.push_back(enc);
    }
    NeuralLMPtr decoder(new NeuralLM(vocab_trg, context_, 0, vm_["wordrep"].as<int>(), vm_["layers"].as<string>(), vocab_trg->GetUnkId(), softmax_sig_, *model));
    encdec.reset(new EncoderDecoder(encoders, decoder, *model));
  }

  BilingualTraining(train_src, train_trg, dev_src, dev_trg,
            *vocab_src, *vocab_trg, *model, *encdec);
}

void LamtramTrain::TrainEncAtt() {

  // cnn::Dict
  DictPtr vocab_trg, vocab_src;
  std::shared_ptr<cnn::Model> model;
  std::shared_ptr<EncoderAttentional> encatt;
  if(model_in_file_.size()) {
    encatt.reset(ModelUtils::LoadBilingualModel<EncoderAttentional>(model_in_file_, model, vocab_src, vocab_trg));
  } else {
    vocab_src.reset(CreateNewDict());
    vocab_trg.reset(CreateNewDict());
    model.reset(new cnn::Model);
  }

  // Read the training file
  vector<Sentence> train_trg, dev_trg, train_src, dev_src;
  LoadFile(train_file_trg_, true, *vocab_trg, train_trg); vocab_trg->Freeze(); vocab_trg->SetUnk("<unk>");
  if(dev_file_trg_.size()) LoadFile(dev_file_trg_, true, *vocab_trg, dev_trg);
  LoadFile(train_file_src_, false, *vocab_src, train_src); vocab_src->Freeze(); vocab_src->SetUnk("<unk>");
  if(dev_file_src_.size()) LoadFile(dev_file_src_, false, *vocab_src, dev_src);
  if(eval_every_ == -1) eval_every_ = train_trg.size();

  // Create the model
  if(model_in_file_.size() == 0) {
    vector<LinearEncoderPtr> encoders;
    vector<string> encoder_types;
    boost::algorithm::split(encoder_types, vm_["encoder_types"].as<string>(), boost::is_any_of("|"));
    for(auto & spec : encoder_types) {
      LinearEncoderPtr enc(new LinearEncoder(vocab_src->size(), vm_["wordrep"].as<int>(), vm_["layers"].as<string>(), vocab_src->GetUnkId(), *model));
      if(spec == "rev") enc->SetReverse(true);
      encoders.push_back(enc);
    }
    BuilderSpec bspec(vm_["layers"].as<string>());
    ExternAttentionalPtr extatt(new ExternAttentional(encoders, vm_["attention_nodes"].as<int>(), bspec.nodes, *model));
    NeuralLMPtr decoder(new NeuralLM(vocab_trg, context_, bspec.nodes * encoders.size(), vm_["wordrep"].as<int>(), vm_["layers"].as<string>(), vocab_trg->GetUnkId(), softmax_sig_, *model));
    encatt.reset(new EncoderAttentional(extatt, decoder, *model));
  }

  BilingualTraining(train_src, train_trg, dev_src, dev_trg,
            *vocab_src, *vocab_trg, *model, *encatt);
}

void LamtramTrain::TrainEncCls() {

  // cnn::Dict
  DictPtr vocab_trg, vocab_src;
  std::shared_ptr<cnn::Model> model;
  std::shared_ptr<EncoderClassifier> enccls;
  if(model_in_file_.size()) {
    enccls.reset(ModelUtils::LoadBilingualModel<EncoderClassifier>(model_in_file_, model, vocab_src, vocab_trg));
  } else {
    vocab_src.reset(CreateNewDict());
    vocab_trg.reset(CreateNewDict());
    model.reset(new cnn::Model);
  }
  // if(!trg_sent) vocab_trg = cnn::Dict("");

  // Read the training files
  vector<int> train_trg, dev_trg;
  vector<Sentence> train_src, dev_src;
  LoadLabels(train_file_trg_, *vocab_trg, train_trg); vocab_trg->Freeze();
  if(dev_file_trg_.size()) LoadLabels(dev_file_trg_, *vocab_trg, dev_trg);
  LoadFile(train_file_src_, false, *vocab_src, train_src); vocab_src->Freeze(); vocab_src->SetUnk("<unk>");
  if(dev_file_src_.size()) LoadFile(dev_file_src_, false, *vocab_src, dev_src);
  if(eval_every_ == -1) eval_every_ = train_trg.size();

  // Create the model
  if(model_in_file_.size() == 0) {
    vector<LinearEncoderPtr> encoders;
    vector<string> encoder_types;
    boost::algorithm::split(encoder_types, vm_["encoder_types"].as<string>(), boost::is_any_of("|"));
    for(auto & spec : encoder_types) {
      LinearEncoderPtr enc(new LinearEncoder(vocab_src->size(), vm_["wordrep"].as<int>(), vm_["layers"].as<string>(), vocab_src->GetUnkId(), *model));
      if(spec == "rev") enc->SetReverse(true);
      encoders.push_back(enc);
    }
    BuilderSpec bspec(vm_["layers"].as<string>());
    ClassifierPtr classifier(new Classifier(bspec.nodes * encoders.size(), vocab_trg->size(), vm_["cls_layers"].as<string>(), *model));
    enccls.reset(new EncoderClassifier(encoders, classifier, *model));
  }

  BilingualTraining(train_src, train_trg, dev_src, dev_trg,
            *vocab_src, *vocab_trg, *model, *enccls);
}

template<class ModelType, class OutputType>
void LamtramTrain::BilingualTraining(const vector<Sentence> & train_src,
                    const vector<OutputType> & train_trg,
                    const vector<Sentence> & dev_src,
                    const vector<OutputType> & dev_trg,
                    const cnn::Dict & vocab_src,
                    const cnn::Dict & vocab_trg,
                    cnn::Model & model,
                    ModelType & encdec) {

  // Sanity checks
  assert(train_src.size() == train_trg.size());
  assert(dev_src.size() == dev_trg.size());

  TrainerPtr trainer = GetTrainer(vm_["trainer"].as<string>(), vm_["learning_rate"].as<double>(), model);
  
  // Learning rate
  cnn::real learning_rate = vm_["learning_rate"].as<double>();
  cnn::real learning_scale = 1.0;

  // Create a sentence list and random generator
  std::vector<int> train_ids(train_src.size());
  std::iota(train_ids.begin(), train_ids.end(), 0);
  // Perform the training
  std::vector<cnn::expr::Expression> empty_hist;
  cnn::real last_ll = -1e99, best_ll = -1e99;
  bool do_dev = dev_src.size() != 0;
  int loc = 0;
  std::shuffle(train_ids.begin(), train_ids.end(), *cnn::rndeng);
  for(int epoch = 0; epoch < epochs_; ++epoch) {
    // Start the training
    LLStats train_ll(vocab_trg.size()), dev_ll(vocab_trg.size());
    Timer time;
    for(auto id_pos : boost::irange(0, eval_every_)) {
      if(loc == (int)train_ids.size()) {
        // Shuffle the access order
        std::shuffle(train_ids.begin(), train_ids.end(), *cnn::rndeng);
        loc = 0;
      }
      cnn::ComputationGraph cg;
      encdec.NewGraph(cg);
      encdec.BuildSentGraph(train_src[train_ids[loc]], train_trg[train_ids[loc]], true, cg, train_ll);
      // cg.PrintGraphviz();
      train_ll.lik_ -= as_scalar(cg.forward());
      cg.backward();
      trainer->update(learning_scale);
      ++loc;
      if(loc % 100 == 0 || id_pos == eval_every_-1 || epochs_ == epoch) {
        double elapsed = time.Elapsed();
        cerr << "Epoch " << epoch+1 << " sent " << loc << ": ppl=" << train_ll.CalcPPL() << ", unk=" << train_ll.unk_ << ", rate=" << learning_scale*learning_rate << ", time=" << elapsed << " (" << train_ll.words_/elapsed << " w/s)" << endl;
        if(epochs_ == epoch) break;
      }
    }
    // Measure development perplexity
    if(do_dev) {
      time = Timer();
      for(int i : boost::irange(0, (int)dev_src.size())) {
        cnn::ComputationGraph cg;
        encdec.NewGraph(cg);
        encdec.BuildSentGraph(dev_src[i], dev_trg[i], false, cg, dev_ll);
        dev_ll.lik_ -= as_scalar(cg.forward());
      }
      double elapsed = time.Elapsed();
      cerr << "Epoch " << epoch+1 << " dev: ppl=" << dev_ll.CalcPPL() << ", unk=" << dev_ll.unk_ << ", rate=" << learning_scale*learning_rate << ", time=" << elapsed << " (" << dev_ll.words_/elapsed << " w/s)" << endl;
    }
    // Adjust the learning rate
    trainer->update_epoch();
    // trainer->status(); cerr << endl;
    // Check the learning rate
    if(last_ll != last_ll)
      THROW_ERROR("Likelihood is not a number, dying...");
    cnn::real my_ll = do_dev ? dev_ll.lik_ : train_ll.lik_;
    if(my_ll < last_ll)
      learning_scale /= 2.0;
    last_ll = my_ll;
    // Open the output stream
    if(best_ll < my_ll) {
      ofstream out(model_out_file_.c_str());
      if(!out) THROW_ERROR("Could not open output file: " << model_out_file_);
      // Write the model (TODO: move this to a separate file?)
      WriteDict(vocab_src, out);
      WriteDict(vocab_trg, out);
      encdec.Write(out);
      ModelUtils::WriteModelText(out, model);
      best_ll = my_ll;
    }
    // If the rate is less than the threshold
    if(learning_scale * learning_rate < rate_thresh_)
      break;
  }
}

void LamtramTrain::LoadFile(const std::string filename, bool add_last, cnn::Dict & vocab, std::vector<Sentence> & sents) {
  ifstream iftrain(filename.c_str());
  if(!iftrain) THROW_ERROR("Could not find training file: " << filename);
  string line;
  int line_no = 0;
  while(getline(iftrain, line)) {
    line_no++;
    Sentence sent = ParseWords(vocab, line, add_last);
    if(sent.size() == (add_last ? 1 : 0))
      THROW_ERROR("Empty line found in " << filename << " at " << line_no << endl);
    sents.push_back(sent);
  }
  iftrain.close();
}

void LamtramTrain::LoadLabels(const std::string filename, cnn::Dict & vocab, std::vector<int> & labs) {
  ifstream iftrain(filename.c_str());
  if(!iftrain) THROW_ERROR("Could not find training file: " << filename);
  string line;
  while(getline(iftrain, line))
    labs.push_back(vocab.Convert(line));
  iftrain.close();
}


LamtramTrain::TrainerPtr LamtramTrain::GetTrainer(const std::string & trainer_id, const cnn::real learning_rate, cnn::Model & model) {
  TrainerPtr trainer;
  if(trainer_id == "sgd") {
    trainer.reset(new cnn::SimpleSGDTrainer(&model, 1e-6, learning_rate));
  } else if(trainer_id == "momentum") {
    trainer.reset(new cnn::MomentumSGDTrainer(&model, 1e-6, learning_rate));
  } else if(trainer_id == "adagrad") {
    trainer.reset(new cnn::AdagradTrainer(&model, 1e-6, learning_rate));
  } else if(trainer_id == "adadelta") {
    trainer.reset(new cnn::AdadeltaTrainer(&model, 1e-6, learning_rate));
  } else if(trainer_id == "adam") {
    trainer.reset(new cnn::AdamTrainer(&model, 1e-6, learning_rate));
  } else {
    THROW_ERROR("Illegal trainer variety: " << trainer_id);
  }
  return trainer;
}
