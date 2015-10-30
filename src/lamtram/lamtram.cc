#include <lamtram/lamtram.h>
#include <lamtram/macros.h>
#include <lamtram/vocabulary.h>
#include <lamtram/sentence.h>
#include <lamtram/timer.h>
#include <lamtram/macros.h>
#include <lamtram/neural-lm.h>
#include <lamtram/encoder-decoder.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/encoder-classifier.h>
#include <lamtram/model-utils.h>
#include <lamtram/ensemble-decoder.h>
#include <lamtram/ensemble-classifier.h>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace lamtram;
namespace po = boost::program_options;

int Lamtram::SequenceOperation(const boost::program_options::variables_map & vm) {
    // Models
    vector<NeuralLMPtr> lms;
    vector<EncoderDecoderPtr> encdecs;
    vector<EncoderAttentionalPtr> encatts;
    vector<shared_ptr<cnn::Model> > models;
    VocabularyPtr vocab_src, vocab_trg;

    // Read in the files
    int pad = 0;
    vector<string> infiles;
    boost::split(infiles, vm["models_in"].as<std::string>(), boost::is_any_of("|"));
    string type, file;
    for(string & infile : infiles) {
        int eqpos = infile.find('=');
        if(eqpos == string::npos)
            THROW_ERROR("Bad model type. Must specify encdec=, encatt=, or nlm= before model name." << endl << infile);
        type = infile.substr(0, eqpos);
        file = infile.substr(eqpos+1);
        VocabularyPtr vocab_src_temp, vocab_trg_temp;
        shared_ptr<cnn::Model> mod_temp;
        // Read in the model
        if(type == "encdec") {
            EncoderDecoder * tm = ModelUtils::LoadModel<EncoderDecoder>(file, mod_temp, vocab_src_temp, vocab_trg_temp);
            encdecs.push_back(shared_ptr<EncoderDecoder>(tm));
            pad = max(pad, tm->GetDecoder().GetNgramContext());
        } else if(type == "encatt") {
            EncoderAttentional * tm = ModelUtils::LoadModel<EncoderAttentional>(file, mod_temp, vocab_src_temp, vocab_trg_temp);
            encatts.push_back(shared_ptr<EncoderAttentional>(tm));
            pad = max(pad, tm->GetDecoder().GetNgramContext());
        } else if(type == "nlm") {
            NeuralLM * lm = ModelUtils::LoadModel<NeuralLM>(file, mod_temp, vocab_src_temp, vocab_trg_temp);
            lms.push_back(shared_ptr<NeuralLM>(lm));
            pad = max(pad, lm->GetNgramContext());
        }
        // Sanity check
        if(vocab_trg.get() && *vocab_trg_temp != *vocab_trg)
            THROW_ERROR("Target vocabularies for translation/language models are not equal.");
        if(vocab_src.get() && vocab_src_temp.get() && *vocab_src_temp != *vocab_src)
            THROW_ERROR("Source vocabularies for translation/language models are not equal.");
        models.push_back(mod_temp);
        vocab_trg = vocab_trg_temp;
        if(vocab_src_temp.get()) vocab_src = vocab_src_temp;
    }
    int vocab_size = vocab_trg->size();

    // Get the source input if necessary
    shared_ptr<ifstream> src_in;
    if(encdecs.size() + encatts.size() > 0) {
        src_in.reset(new ifstream(vm["src_in"].as<std::string>()));
        if(!*src_in)
            THROW_ERROR("Could not find src_in file " << vm["src_in"].as<std::string>());
    }
    
    // Create the decoder
    EnsembleDecoder decoder(encdecs, encatts, lms, pad);
    decoder.SetWordPen(vm["word_pen"].as<float>());
    decoder.SetEnsembleOperation(vm["ensemble_op"].as<string>());
    decoder.SetBeamSize(vm["beam"].as<int>());
    
    // Perform operation
    string operation = vm["operation"].as<std::string>();
    string line;
    Sentence sent_src, sent_trg;
    if(operation == "ppl") {
        LLStats corpus_ll(vocab_size);
        Timer time;
        while(getline(cin, line)) { 
            LLStats sent_ll(vocab_size);
            // Get the target, and if it exists, source sentences
            if(GlobalVars::verbose >= 2) { cerr << "SentLL trg: " << line << endl; }
            sent_trg = vocab_trg->ParseWords(line, pad, true);
            if(encdecs.size() + encatts.size() > 0) {
                if(!getline(*src_in, line))
                    THROW_ERROR("Source and target files don't match");
                if(GlobalVars::verbose >= 2) { cerr << "SentLL src: " << line << endl; }
                sent_src = vocab_src->ParseWords(line, 0, false);
            }
            // If the encoder
            decoder.CalcSentLL(sent_src, sent_trg, sent_ll);
            if(GlobalVars::verbose >= 1) { cout << "ll=" << sent_ll.CalcUnkLik() << " unk=" << sent_ll.unk_  << endl; }
            corpus_ll += sent_ll;
        }
        double elapsed = time.Elapsed();
        cerr << "ppl=" << corpus_ll.CalcPPL() << ", unk=" << corpus_ll.unk_ << ", time=" << elapsed << " (" << corpus_ll.words_/elapsed << " w/s)" << endl;
    } else if(operation == "gen") {
        int max_sents = vm["sents"].as<int>();
        if(max_sents == 0) max_sents = INT_MAX;
        while(max_sents-- > 0) {
            if(encdecs.size() + encatts.size() > 0) {
                if(!getline(*src_in, line)) break;
                sent_src = vocab_src->ParseWords(line, 0, false);
            }
            sent_trg = decoder.Generate(sent_src);
            cout << vocab_trg->PrintWords(sent_trg, false) << endl;
        }
    } else {
        THROW_ERROR("Illegal operation " << operation);
    }

    return 0;
}

int Lamtram::ClassifierOperation(const boost::program_options::variables_map & vm) {
    // Models
    vector<EncoderClassifierPtr> encclss;
    shared_ptr<Vocabulary> vocab_src, vocab_trg;
    vector<shared_ptr<cnn::Model> > models;

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
        VocabularyPtr vocab_src_temp, vocab_trg_temp;
        shared_ptr<cnn::Model> mod_temp;
        // Read in the model
        EncoderClassifier * tm = ModelUtils::LoadModel<EncoderClassifier>(file, mod_temp, vocab_src_temp, vocab_trg_temp);
        encclss.push_back(shared_ptr<EncoderClassifier>(tm));
        // Sanity check
        if(vocab_trg.get() && *vocab_trg_temp != *vocab_trg)
            THROW_ERROR("Target vocabularies for translation/language models are not equal.");
        if(vocab_src.get() && vocab_src_temp.get() && *vocab_src_temp != *vocab_src)
            THROW_ERROR("Target vocabularies for translation/language models are not equal.");
        models.push_back(mod_temp);
        vocab_trg = vocab_trg_temp;
        vocab_src = vocab_src_temp;
    }
    int vocab_size = vocab_trg->size();

    // Get the source input
    shared_ptr<ifstream> src_in;
    src_in.reset(new ifstream(vm["src_in"].as<std::string>()));
    if(!*src_in)
        THROW_ERROR("Could not find src_in file " << vm["src_in"].as<std::string>());
    
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
            trg = vocab_trg->WID(line);
            if(!getline(*src_in, line))
                THROW_ERROR("Source and target files don't match");
            if(GlobalVars::verbose > 0) { cerr << "ClsEval src: " << line << endl; }
            sent_src = vocab_src->ParseWords(line, 0, false);
            // If the encoder
            ensemble.CalcEval(sent_src, trg, sent_ll);
            if(GlobalVars::verbose > 0) { cout << "ll=" << sent_ll.CalcUnkLik() << " correct=" << sent_ll.correct_ << endl; }
            corpus_ll += sent_ll;
        }
        double elapsed = time.Elapsed();
        cerr << "ppl=" << corpus_ll.CalcPPL() << ", acc="<< corpus_ll.CalcAcc() << ", time=" << elapsed << " (" << corpus_ll.words_/elapsed << " w/s)" << endl;
    } else if(operation == "cls") {
        while(getline(*src_in, line)) {
            sent_src = vocab_src->ParseWords(line, 0, false);
            trg = ensemble.Predict(sent_src);
            cout << vocab_trg->WSym(trg) << endl;
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
        ("operation", po::value<string>()->default_value("ppl"), "Operations (ppl: measure perplexity, gen: generate sentences)")
        ("models_in", po::value<string>()->default_value(""), "Model files in format \"{encdec,encatt,nlm}=filename\" with encdec for encoder-decoders, encatt for attentional models, nlm for language models. When multiple, separate by a pipe.")
        ("src_in", po::value<string>()->default_value(""), "File to read the source from, if any")
        ("word_pen", po::value<float>()->default_value(0.0), "The \"word penalty\", a larger value favors longer sentences, shorter favors shorter")
        ("ensemble_op", po::value<string>()->default_value("sum"), "The operation to use when ensembling probabilities (sum/logsum)")
        ("beam", po::value<int>()->default_value(1), "Number of hypotheses to expand")
        ("sents", po::value<int>()->default_value(0), "When generating, maximum of how many sentences (0 for no limit)")
        ("verbose", po::value<int>()->default_value(0), "How much verbose output to print")
        ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);   
    if (vm.count("help")) {
        cout << desc << endl;
        return 1;
    }

    GlobalVars::verbose = vm["verbose"].as<int>();

    string operation = vm["operation"].as<std::string>();
    if(operation == "ppl" || operation == "gen") {
        return SequenceOperation(vm);
    } else if(operation == "cls" || operation == "clseval") {
        return ClassifierOperation(vm);
    } else {
        THROW_ERROR("Illegal operation: " << operation);
    }

}
