#include <lamtram/lamtram.h>
#include <lamtram/macros.h>
#include <lamtram/sentence.h>
#include <lamtram/timer.h>
#include <lamtram/macros.h>
#include <lamtram/neural-lm.h>
#include <lamtram/encoder-decoder.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/encoder-classifier.h>
#include <lamtram/model-utils.h>
#include <lamtram/string-util.h>
#include <lamtram/ensemble-decoder.h>
#include <lamtram/ensemble-classifier.h>
#include <lamtram/mapping.h>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <dynet/dict.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace lamtram;
namespace po = boost::program_options;

// typedef std::vector<std::vector<float> > Sentence;

void Lamtram::MapWords(const vector<string> & src_strs, const Sentence & trg_sent, const Sentence & align, const UniqueStringMappingPtr & mapping, vector<string> & trg_strs) {
  if(align.size() == 0) return;
  assert(trg_sent.size() >= trg_strs.size());
  assert(align.size() == trg_sent.size());
  WordId unk_id = 1;
  for(size_t i = 0; i < trg_strs.size(); i++) {
    if(trg_sent[i] == unk_id) {
      size_t max_id = align[i];
      if(max_id != -1) {
        if(src_strs.size() <= max_id) {
          trg_strs[i] = "<unk>";
        } else if(mapping.get() != nullptr) {
          auto it = mapping->find(src_strs[max_id]);
          trg_strs[i] = (it != mapping->end()) ? it->second.first : src_strs[max_id];
        } else {
          trg_strs[i] = src_strs[max_id];
        }
      }
    }
  }
}

int Lamtram::SequenceOperation(const boost::program_options::variables_map & vm) {
  // Models
  vector<NeuralLMPtr> lms;
  vector<EncoderDecoderPtr> encdecs;
  vector<EncoderAttentionalPtr> encatts;
  vector<shared_ptr<dynet::Model> > models;
  DictPtr vocab_src, vocab_trg;

  int max_minibatch_size = vm["minibatch_size"].as<int>();
  int nbest_size = vm["nbest_size"].as<int>();
  
  // Buffers
  string line;
  vector<string> strs;

  // Read in the files
  vector<string> infiles;
  boost::split(infiles, vm["models_in"].as<std::string>(), boost::is_any_of("|"));
  string type, file;
  for(string & infile : infiles) {
    int eqpos = infile.find('=');
    if(eqpos == string::npos)
      THROW_ERROR("Bad model type. Must specify encdec=, encatt=, or nlm= before model name." << endl << infile);
    type = infile.substr(0, eqpos);
    file = infile.substr(eqpos+1);
    DictPtr vocab_src_temp, vocab_trg_temp;
    shared_ptr<dynet::Model> mod_temp;
    // Read in the model
    if(type == "encdec") {
      EncoderDecoder * tm = ModelUtils::LoadBilingualModel<EncoderDecoder>(file, mod_temp, vocab_src_temp, vocab_trg_temp);
      encdecs.push_back(shared_ptr<EncoderDecoder>(tm));
    } else if(type == "encatt") {
      EncoderAttentional * tm = ModelUtils::LoadBilingualModel<EncoderAttentional>(file, mod_temp, vocab_src_temp, vocab_trg_temp);
      encatts.push_back(shared_ptr<EncoderAttentional>(tm));
    } else if(type == "nlm") {
      NeuralLM * lm = ModelUtils::LoadMonolingualModel<NeuralLM>(file, mod_temp, vocab_trg_temp);
      lms.push_back(shared_ptr<NeuralLM>(lm));
    }
    // Sanity check
    if(vocab_trg.get() && vocab_trg_temp->get_words() != vocab_trg->get_words())
      THROW_ERROR("Target vocabularies for translation/language models are not equal.");
    if(vocab_src.get() && vocab_src_temp.get() && vocab_src_temp->get_words() != vocab_src->get_words())
      THROW_ERROR("Source vocabularies for translation/language models are not equal.");
    models.push_back(mod_temp);
    vocab_trg = vocab_trg_temp;
    if(vocab_src_temp.get()) vocab_src = vocab_src_temp;
  }
  int vocab_size = vocab_trg->size();

  // Get the mapping table if necessary
  UniqueStringMappingPtr mapping;
  if(vm["map_in"].as<std::string>() != "")
    mapping.reset(LoadUniqueStringMapping(vm["map_in"].as<std::string>()));

  // Get the source input if necessary, "-" means stdin
  shared_ptr<ifstream> src_in_sptr;
  istream* src_in = NULL;
  if(encdecs.size() + encatts.size() > 0) {
    string src_in_path = vm["src_in"].as<std::string>();
    if (src_in_path == "-") {
      cerr << "Reading from stdin" << endl;
      src_in = &cin;
    } else {
      src_in_sptr.reset(new ifstream(src_in_path));
      if (!*src_in_sptr)
        THROW_ERROR("Could not find src_in file " << src_in_path);
      src_in = src_in_sptr.get();
    }
  }
  
  // Find the range
  pair<size_t,size_t> sent_range(0,INT_MAX);
  if(vm["sent_range"].as<string>() != "") {
    std::vector<string> range_str = Tokenize(vm["sent_range"].as<string>(), ",");
    if(range_str.size() != 2)
      THROW_ERROR("When specifying a range must be two comma-delimited numbers, but got: " << vm["sent_range"].as<string>());
    sent_range.first = std::stoi(range_str[0]);
    sent_range.second = std::stoi(range_str[1]);
  }
  
  // Create the decoder
  EnsembleDecoder decoder(encdecs, encatts, lms);
  decoder.SetWordPen(vm["word_pen"].as<float>());
  decoder.SetUnkPen(vm["unk_pen"].as<float>());
  decoder.SetEnsembleOperation(vm["ensemble_op"].as<string>());
  decoder.SetBeamSize(vm["beam"].as<int>());
  decoder.SetSizeLimit(vm["max_len"].as<int>());

  
  // Perform operation
  string operation = vm["operation"].as<std::string>();
  string wpout_file = vm["wordprob_out"].as<std::string>();
  Sentence sent_src, sent_trg;
  vector<string> str_src, str_trg;
  Sentence align;
  int last_id = -1;
  bool do_sent = false;
  if(operation == "ppl") {
    shared_ptr<ofstream> wpout;
    if(wpout_file != "")
      wpout.reset(new ofstream(wpout_file));
    LLStats corpus_ll(vocab_size);
    Timer time;
    while(getline(cin, line)) { 
      // Get the target, and if it exists, source sentences
      if(GlobalVars::verbose >= 2) { cerr << "SentLL trg: " << line << endl; }
      sent_trg = ParseWords(*vocab_trg, line, true);
      if(encdecs.size() + encatts.size() > 0) {
        if(!getline(*src_in, line))
          THROW_ERROR("Source and target files don't match");
        if(GlobalVars::verbose >= 2) { cerr << "SentLL src: " << line << endl; }
        sent_src = ParseWords(*vocab_src, line, false);
      }
      last_id++;
      // If we're inside the range, do it
      if(last_id >= sent_range.first && last_id < sent_range.second) {
        LLStats sent_ll(vocab_size);
        vector<float> word_lls;
        decoder.CalcSentLL<Sentence,LLStats,vector<float> >(sent_src, sent_trg, sent_ll, word_lls);
        if(GlobalVars::verbose >= 1) { cout << "ll=" << -sent_ll.CalcUnkLoss() << " unk=" << sent_ll.unk_  << endl; }
        corpus_ll += sent_ll;
        // Write word probabilities if necessary
        if(wpout.get()) {
          if(word_lls.size()) *wpout << -word_lls[0];
          for(size_t i = 1; i < word_lls.size(); i++) *wpout << ' ' << -word_lls[i];
          *wpout << endl;
        }
      }
    }
    double elapsed = time.Elapsed();
    cerr << "ppl=" << corpus_ll.CalcPPL() << ", unk=" << corpus_ll.unk_ << ", time=" << elapsed << " (" << corpus_ll.words_/elapsed << " w/s)" << endl;
  } else if(operation == "nbest") {
    Timer time;
    int all_words = 0, curr_words = 0;
    std::vector<Sentence> sents_trg;
    while(getline(cin, line)) { 
      // Get the new sentence
      vector<string> columns = Tokenize(line, " ||| ");
      if(columns.size() < 2) THROW_ERROR("Bad line in n-best:\n" << line);
      int my_id = stoi(columns[0]);
      sent_trg = ParseWords(*vocab_trg, columns[1], true);
      // If we've finished the current source, print
      if((my_id != last_id || curr_words+sents_trg.size() > max_minibatch_size) && sents_trg.size() > 0) {
        vector<LLStats> sents_ll(sents_trg.size(), LLStats(vocab_size));
        vector<vector<float> > word_lls(sents_trg.size());
        if(sents_trg.size() > 1)
          decoder.CalcSentLL<vector<Sentence>,vector<LLStats>,vector<vector<float> > >(sent_src, sents_trg, sents_ll, word_lls);
        else
          decoder.CalcSentLL<Sentence,LLStats,vector<float> >(sent_src, sents_trg[0], sents_ll[0], word_lls[0]);
        for(auto & sent_ll : sents_ll)
          cout << "ll=" << -sent_ll.CalcUnkLoss() << " unk=" << sent_ll.unk_  << endl;
        sents_trg.resize(0);
        curr_words = 0;
      }
      // Load the new source word
      if(my_id != last_id) {
        if(!getline(*src_in, line))
          THROW_ERROR("Source and target files don't match");
        sent_src = ParseWords(*vocab_src, line, false);
        if(do_sent) {
          double elapsed = time.Elapsed();
          cerr << "sent=" << last_id << ", time=" << elapsed << " (" << all_words/elapsed << " w/s)" << endl;
        }
        last_id = my_id;
        do_sent = (last_id >= sent_range.first && last_id < sent_range.second);
      }
      // Add to the data
      if(do_sent) {
        sents_trg.push_back(sent_trg);
        all_words += sent_trg.size();
        curr_words += sent_trg.size();
      }
    }
    if(do_sent) {
      vector<LLStats> sents_ll(sents_trg.size(), LLStats(vocab_size));
      vector<vector<float> > word_lls;
      decoder.CalcSentLL<vector<Sentence>,vector<LLStats>, vector<vector<float> > >(sent_src, sents_trg, sents_ll, word_lls);
      for(auto & sent_ll : sents_ll)
        cout << "ll=" << -sent_ll.CalcUnkLoss() << " unk=" << sent_ll.unk_  << endl;
      double elapsed = time.Elapsed();
      cerr << "sent=" << last_id << ", time=" << elapsed << " (" << all_words/elapsed << " w/s)" << endl;
    }
  } else if(operation == "gen" || operation == "samp") {
    if(operation == "samp") THROW_ERROR("Sampling not implemented yet");
    for(int i = 0; i < sent_range.second; ++i) {
      if(encdecs.size() + encatts.size() > 0) {
        if(!getline(*src_in, line)) break;
        str_src = SplitWords(line);
        sent_src = ParseWords(*vocab_src, str_src, false);
      }
      if(i >= sent_range.first) {
        if(nbest_size == 1) {
          EnsembleDecoderHypPtr trg_hyp = decoder.Generate(sent_src);
          if(trg_hyp.get() == nullptr) {
            cout << endl;
          } else {
            sent_trg = trg_hyp->GetSentence();
            align = trg_hyp->GetAlignment();
            str_trg = ConvertWords(*vocab_trg, sent_trg, false);
            MapWords(str_src, sent_trg, align, mapping, str_trg);
            cout << PrintWords(str_trg) << endl;
          }
        } else {
          auto trg_hyps = decoder.GenerateNbest(sent_src, nbest_size);
          for(auto & trg_hyp : trg_hyps) {
            if(trg_hyp.get() != nullptr) {
              sent_trg = trg_hyp->GetSentence();
              align = trg_hyp->GetAlignment();
              str_trg = ConvertWords(*vocab_trg, sent_trg, false);
              MapWords(str_src, sent_trg, align, mapping, str_trg);
              cout << i << " ||| " << PrintWords(str_trg) << " ||| " << trg_hyp->GetScore() << endl;
            }
          }
        }
      }
    }
  } else {
    THROW_ERROR("Illegal operation " << operation);
  }

  return 0;
}

int Lamtram::ClassifierOperation(const boost::program_options::variables_map & vm) {
  // Models
  vector<EncoderClassifierPtr> encclss;
  DictPtr vocab_src, vocab_trg;
  vector<shared_ptr<dynet::Model> > models;

  // Read in the files
  vector<string> infiles;
  boost::split(infiles, vm["models_in"].as<std::string>(), boost::is_any_of("|"));
  string type, file;
  for(string & infile : infiles) {
    int eqpos = infile.find('=');
    if(eqpos == string::npos)
      THROW_ERROR("Bad model type. Must specify enccls= before model name." << endl << infile);
    type = infile.substr(0, eqpos);
    if(type != "enccls")
      THROW_ERROR("Bad model type. Must specify enccls= before model name." << endl << infile);
    file = infile.substr(eqpos+1);
    DictPtr vocab_src_temp, vocab_trg_temp;
    shared_ptr<dynet::Model> mod_temp;
    // Read in the model
    EncoderClassifier * tm = ModelUtils::LoadBilingualModel<EncoderClassifier>(file, mod_temp, vocab_src_temp, vocab_trg_temp);
    encclss.push_back(shared_ptr<EncoderClassifier>(tm));
    // Sanity check
    if(vocab_trg.get() && vocab_trg_temp->get_words() != vocab_trg->get_words())
      THROW_ERROR("Target vocabularies for translation/language models are not equal.");
    if(vocab_src.get() && vocab_src_temp.get() && vocab_src_temp->get_words() != vocab_src->get_words())
      THROW_ERROR("Target vocabularies for translation/language models are not equal.");
    models.push_back(mod_temp);
    vocab_trg = vocab_trg_temp;
    vocab_src = vocab_src_temp;
  }
  int vocab_size = vocab_trg->size();

  // Get the source input if necessary, "-" means stdin
  shared_ptr<ifstream> src_in_sptr;
  istream* src_in = NULL;
  string src_in_path = vm["src_in"].as<std::string>();
  if (src_in_path == "-") {
    cerr << "Reading from stdin" << endl;
    src_in = &cin;
  } else {
    src_in_sptr.reset(new ifstream(src_in_path));
    if (!*src_in_sptr)
      THROW_ERROR("Could not find src_in file " << src_in_path);
    src_in = src_in_sptr.get();
  }

  // Create the decoder
  EnsembleClassifier ensemble(encclss);
  ensemble.SetEnsembleOperation(vm["ensemble_op"].as<string>());
  
  // Perform operation
  string operation = vm["operation"].as<std::string>();
  string line;
  Sentence sent_src;
  int trg;
  if(operation == "clseval") {
    LLStats corpus_ll(vocab_size);
    Timer time;
    while(getline(cin, line)) { 
      LLStats sent_ll(vocab_size);
      // Get the target, and if it exists, source sentences
      if(GlobalVars::verbose > 0) { cerr << "ClsEval trg: " << line << endl; }
      trg = vocab_trg->convert(line);
      if(!getline(*src_in, line))
        THROW_ERROR("Source and target files don't match");
      if(GlobalVars::verbose > 0) { cerr << "ClsEval src: " << line << endl; }
      sent_src = ParseWords(*vocab_src, line, false);
      // If the encoder
      ensemble.CalcEval(sent_src, trg, sent_ll);
      if(GlobalVars::verbose > 0) { cout << "ll=" << -sent_ll.CalcUnkLoss() << " correct=" << sent_ll.correct_ << endl; }
      corpus_ll += sent_ll;
    }
    double elapsed = time.Elapsed();
    cerr << "ppl=" << corpus_ll.CalcPPL() << ", acc="<< corpus_ll.CalcAcc() << ", time=" << elapsed << " (" << corpus_ll.words_/elapsed << " w/s)" << endl;
  } else if(operation == "cls") {
    while(getline(*src_in, line)) {
      sent_src = ParseWords(*vocab_src, line, false);
      trg = ensemble.Predict(sent_src);
      cout << vocab_trg->convert(trg) << endl;
    }
  } else {
    THROW_ERROR("Illegal operation " << operation);
  }

  return 0;
}

int Lamtram::main(int argc, char** argv) {
  po::options_description desc("*** lamtram-train (by Graham Neubig) ***");
  desc.add_options()
    ("help", "Produce help message")
    ("verbose", po::value<int>()->default_value(0), "How much verbose output to print")
    ("beam", po::value<int>()->default_value(1), "Number of hypotheses to expand")
    ("dynet_mem", po::value<int>()->default_value(512), "How much memory to allocate to dynet")
    ("ensemble_op", po::value<string>()->default_value("sum"), "The operation to use when ensembling probabilities (sum/logsum)")
    ("wordprob_out", po::value<string>()->default_value(""), "Output word log probabilities during perplexity calculation")
    ("map_in", po::value<string>()->default_value(""), "A file containing a mapping table (\"src trg prob\" format)")
    ("minibatch_size", po::value<int>()->default_value(1), "Max size of a minibatch in words (may be exceeded if there are longer sentences)")
    ("models_in", po::value<string>()->default_value(""), "Model files in format \"{encdec,encatt,nlm}=filename\" with encdec for encoder-decoders, encatt for attentional models, nlm for language models. When multiple, separate by a pipe.")
    ("nbest_size", po::value<int>()->default_value(1), "The size of an n-best to generate when generating n-best")
    ("operation", po::value<string>()->default_value("ppl"), "Operations (ppl: measure perplexity, nbest: score n-best list, gen: generate most likely sentence, samp: sample sentences randomly)")
    ("sent_range", po::value<string>()->default_value(""), "Optionally specify a comma-delimited range on how many sentences to process")
    ("max_len", po::value<int>()->default_value(200), "Limit on the max length of sentences")
    ("src_in", po::value<string>()->default_value("-"), "File to read the source from, if any")
    ("word_pen", po::value<float>()->default_value(0.f), "The \"word penalty\", a larger value favors longer sentences, shorter favors shorter")
    ("unk_pen", po::value<float>()->default_value(0.f), "A penalty for unknown words, larger will create fewer unknown words when decoding")
    ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);   
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  for(int i = 0; i < argc; i++) { cerr << argv[i] << " "; } cerr << endl;

  GlobalVars::verbose = vm["verbose"].as<int>();

  string operation = vm["operation"].as<std::string>();
  if(operation == "ppl" || operation == "nbest" || operation == "gen" || operation == "samp") {
    return SequenceOperation(vm);
  } else if(operation == "cls" || operation == "clseval") {
    return ClassifierOperation(vm);
  } else {
    THROW_ERROR("Illegal operation: " << operation);
  }

}
