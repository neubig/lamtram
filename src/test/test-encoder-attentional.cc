#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <fstream>

#include <dynet/dict.h>
#include <dynet/training.h>

#include <lamtram/macros.h>
#include <lamtram/encoder-decoder.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/ensemble-decoder.h>
#include <lamtram/model-utils.h>

using namespace std;
using namespace lamtram;

// ****** The fixture *******
struct TestEncoderAttentional {

  TestEncoderAttentional() : sent_src_(4), sent_trg_(4) {
    sent_src_ = {1, 2, 3, 0};
    sent_trg_ = {3, 2, 1, 0};
    sent_src2_ = {4, 3, 2, 0};
    sent_trg2_ = {1, 4, 3, 0};

    // Create the vocab
    vocab_src_ = DictPtr(CreateNewDict()); vocab_src_->convert("a"); vocab_src_->convert("b"); vocab_src_->convert("c");
    vocab_trg_ = DictPtr(CreateNewDict()); vocab_trg_->convert("x"); vocab_trg_->convert("y"); vocab_trg_->convert("z");

  }
  ~TestEncoderAttentional() { }

  void CreateModel(
        shared_ptr<dynet::Model> & mod,
        EncoderAttentionalPtr & encatt,
        shared_ptr<EnsembleDecoder> & ensdec,
        const std::string & attention_type = "mlp:2",
        bool attention_feed = false,
        const std::string & attention_hist = "none",
        const std::string & lex_type = "none"
  ) {
    // Create a dummy lexicon file if necessary
    string my_lex_type = lex_type;
    if(lex_type == "prior") {
      ofstream ofs("/tmp/lex_prior.txt");
      if(!ofs) THROW_ERROR("Could not open /tmp/lex_prior.txt for writing");
      ofs << "a\tx\t0.7\na\ty\t0.3\nb\tz\t1.0\nc\ty\t1.0\n";
      my_lex_type = "prior:file=/tmp/lex_prior.txt:alpha=0.001";
    }
    // Create the model
    mod = shared_ptr<dynet::Model>(new dynet::Model);
    NeuralLMPtr lmptr(new NeuralLM(vocab_trg_, 1, (attention_feed ? 5 : 0), attention_feed, 5, BuilderSpec("lstm:5:1"), -1, "full", *mod));
    vector<LinearEncoderPtr> encs(1, LinearEncoderPtr(new LinearEncoder(vocab_src_->size(), 5, BuilderSpec("lstm:5:1"), -1, *mod)));
    ExternAttentionalPtr ext(new ExternAttentional(encs, attention_type, attention_hist, 5, my_lex_type, vocab_src_, vocab_trg_, *mod));
    encatt = shared_ptr<EncoderAttentional>(new EncoderAttentional(ext, lmptr, *mod));
    // Create the ensemble decoder
    vector<EncoderDecoderPtr> encdecs;
    vector<EncoderAttentionalPtr> encatts; encatts.push_back(encatt);
    vector<NeuralLMPtr> lms;
    ensdec = shared_ptr<EnsembleDecoder>(new EnsembleDecoder(encdecs, encatts, lms));
    ensdec->SetSizeLimit(100);
    if(lex_type == "prior")
      std::remove("/tmp/lex_prior.txt");
    // Perform a few rounds of training
    dynet::SimpleSGDTrainer sgd(*mod);
    LLStats train_stat(vocab_trg_->size());
    for(size_t i = 0; i < 100; ++i) {
      dynet::ComputationGraph cg;
      encatt->NewGraph(cg);
      dynet::expr::Expression loss_expr = encatt->BuildSentGraph(sent_src_, sent_trg_, cache_, nullptr, 0.f, false, cg, train_stat);
      cg.forward(loss_expr);
      cg.backward(loss_expr);
      sgd.update(0.1);
    }
  }

  void TestLLScores(
     const std::string & attention_type,
     bool attention_feed,
     const std::string & attention_hist,
     const std::string & lex_type
  ) {
    shared_ptr<dynet::Model> mod;
    EncoderAttentionalPtr encatt;
    shared_ptr<EnsembleDecoder> ensdec;
    CreateModel(mod, encatt, ensdec, attention_type, attention_feed, attention_hist, lex_type);
    // Compare the two values
    LLStats train_stat(vocab_trg_->size()), test_stat(vocab_trg_->size());
    {
      dynet::ComputationGraph cg;
      encatt->NewGraph(cg);
      dynet::expr::Expression loss_expr = encatt->BuildSentGraph(sent_src_, sent_trg_, cache_, nullptr, 0.f, false, cg, train_stat);
      train_stat.loss_ += as_scalar(cg.incremental_forward(loss_expr));
    }
    vector<float> test_wordll;
    ensdec->CalcSentLL(sent_src_, sent_trg_, test_stat, test_wordll);
    BOOST_CHECK_CLOSE(train_stat.CalcPPL(), test_stat.CalcPPL(), 0.01);
  }

  void TestDecoding(const std::string & attention_type, bool attention_feed, const std::string & attention_hist, const std::string & lex_type) {
    shared_ptr<dynet::Model> mod;
    EncoderAttentionalPtr encatt;
    shared_ptr<EnsembleDecoder> ensdec;
    CreateModel(mod, encatt, ensdec, attention_type, attention_feed, attention_hist, lex_type);
    // Train
    float train_ll = 0, decode_ll = 0;
    Sentence decode_sent;
    // Perform decoding
    {
      vector<EnsembleDecoderHypPtr> hyps = ensdec->GenerateNbest(sent_src_, 1);
      decode_ll = hyps[0]->GetScore();
      decode_sent = hyps[0]->GetSentence();
    }
    // Calculate the training likelihood for that value
    {
      LLStats train_stat(vocab_trg_->size());
      dynet::ComputationGraph cg;
      encatt->NewGraph(cg);
      dynet::expr::Expression loss_expr = encatt->BuildSentGraph(sent_src_, decode_sent, cache_, nullptr, 0.f, false, cg, train_stat);
      train_ll = -as_scalar(cg.incremental_forward(loss_expr));
    }
    BOOST_CHECK_CLOSE(train_ll, decode_ll, 0.01);
  }

  Sentence sent_src_, sent_trg_, sent_src2_, sent_trg2_, cache_;
  DictPtr vocab_src_, vocab_trg_;
};

// ****** The tests *******
BOOST_FIXTURE_TEST_SUITE(encoder_attentional, TestEncoderAttentional)

// Test whether reading and writing works.
// Note that this is just checking if serialized strings is equal,
// which is a hack for now because dynet::Model doesn't have an equality operator.
BOOST_AUTO_TEST_CASE(TestWriteRead) {
  // Create a randomized lm
  shared_ptr<dynet::Model> act_mod(new dynet::Model), exp_mod(new dynet::Model);
  DictPtr exp_src_vocab(CreateNewDict()); exp_src_vocab->convert("hola");
  DictPtr exp_trg_vocab(CreateNewDict()); exp_trg_vocab->convert("hello");
  NeuralLMPtr exp_lm(new NeuralLM(exp_trg_vocab, 2, 2, false, 3, BuilderSpec("rnn:2:1"), -1, "full", *exp_mod));
  vector<LinearEncoderPtr> exp_encs(1, LinearEncoderPtr(new LinearEncoder(3, 2, BuilderSpec("rnn:2:1"), -1, *exp_mod)));
  ExternAttentionalPtr exp_ext(new ExternAttentional(exp_encs, "mlp:2", "none", 3, "none", vocab_src_, vocab_trg_, *exp_mod));
  EncoderAttentional exp_encatt(exp_ext, exp_lm, *exp_mod);
  // Write the Model
  ostringstream out;
  WriteDict(*exp_src_vocab, out);
  WriteDict(*exp_trg_vocab, out);
  exp_encatt.Write(out);
  ModelUtils::WriteModelText(out, *exp_mod);
  // Read the Model
  DictPtr act_src_vocab(new dynet::Dict), act_trg_vocab(new dynet::Dict);
  string first_string = out.str();
  istringstream in(out.str());
  EncoderAttentionalPtr act_lm(ModelUtils::LoadBilingualModel<EncoderAttentional>(in, act_mod, act_src_vocab, act_trg_vocab));
  // Write to a second string
  ostringstream out2;
  WriteDict(*act_src_vocab, out2);
  WriteDict(*act_trg_vocab, out2);
  act_lm->Write(out2);
  ModelUtils::WriteModelText(out2, *act_mod);
  string second_string = out2.str();
  // Check if the two
  BOOST_CHECK_EQUAL(first_string, second_string);
}

// Test whether scores during likelihood calculation are the same as training
BOOST_AUTO_TEST_CASE(TestLLScoresDotFalseNone)      { TestLLScores("dot",   false, "none", "none"); }
BOOST_AUTO_TEST_CASE(TestLLScoresDotFalseNonePrior) { TestLLScores("dot",   false, "none", "prior"); }
BOOST_AUTO_TEST_CASE(TestLLScoresDotTrueNone)       { TestLLScores("dot",   true,  "none", "none"); }
BOOST_AUTO_TEST_CASE(TestLLScoresDotFalseSum)       { TestLLScores("dot",   false, "sum" , "none"); }
BOOST_AUTO_TEST_CASE(TestLLScoresMLPFalseNone)      { TestLLScores("mlp:5", false, "none", "none"); }
BOOST_AUTO_TEST_CASE(TestLLScoresMLPTrueSum)        { TestLLScores("mlp:5", true,  "sum" , "none"); }
BOOST_AUTO_TEST_CASE(TestLLScoresBilinFalseNone)    { TestLLScores("bilin", false, "none", "none"); }


// Test whether log likelihood is the same when batched or not
BOOST_AUTO_TEST_CASE(TestLLBatchScores) {
  shared_ptr<dynet::Model> mod;
  EncoderAttentionalPtr encatt;
  shared_ptr<EnsembleDecoder> ensdec;
  CreateModel(mod, encatt, ensdec, "mlp:5", true, "sum");
  LLStats batch_stat(vocab_trg_->size()), unbatch_stat(vocab_trg_->size());
  // Do unbatched calculation
  {
    dynet::ComputationGraph cg; encatt->NewGraph(cg);
    dynet::expr::Expression loss_expr = encatt->BuildSentGraph(sent_src_, sent_trg_, cache_, nullptr, 0.f, false, cg, unbatch_stat);
    unbatch_stat.loss_ += as_scalar(cg.incremental_forward(loss_expr));
  }
  {
    dynet::ComputationGraph cg; encatt->NewGraph(cg);
    dynet::expr::Expression loss_expr = encatt->BuildSentGraph(sent_src2_, sent_trg2_, cache_, nullptr, 0.f, false, cg, unbatch_stat);
    unbatch_stat.loss_ += as_scalar(cg.incremental_forward(loss_expr));
  }
  // Do batched calculation
  {
    std::vector<Sentence> batch_src(2); batch_src[0] = sent_src_; batch_src[1] = sent_src2_;
    std::vector<Sentence> batch_trg(2); batch_trg[0] = sent_trg_; batch_trg[1] = sent_trg2_;
    std::vector<Sentence> batch_cache(2); batch_cache[0] = cache_; batch_cache[1] = cache_;
    dynet::ComputationGraph cg; encatt->NewGraph(cg);
    dynet::expr::Expression loss_expr = encatt->BuildSentGraph(batch_src, batch_trg, batch_cache, nullptr, 0.f, false, cg, batch_stat);
    batch_stat.loss_ += as_scalar(cg.incremental_forward(loss_expr));
  }
  BOOST_CHECK_CLOSE(unbatch_stat.CalcPPL(), batch_stat.CalcPPL(), 0.5);
}

// Test whether scores during decoding are the same as training
BOOST_AUTO_TEST_CASE(TestDecodingDotFalseNone)      { TestDecoding("dot",   false, "none", "none"); }
BOOST_AUTO_TEST_CASE(TestDecodingDotFalseNonePrior) { TestDecoding("dot",   false, "none", "prior"); }
BOOST_AUTO_TEST_CASE(TestDecodingDotTrueNone)       { TestDecoding("dot",   true,  "none", "none"); }
BOOST_AUTO_TEST_CASE(TestDecodingDotFalseSum)       { TestDecoding("dot",   false, "sum" , "none"); }
BOOST_AUTO_TEST_CASE(TestDecodingMLPFalseNone)      { TestDecoding("mlp:5", false, "none", "none"); }
BOOST_AUTO_TEST_CASE(TestDecodingMLPTrueSum)        { TestDecoding("mlp:5", true,  "sum" , "none"); }
BOOST_AUTO_TEST_CASE(TestDecodingBilinFalseNone)    { TestDecoding("bilin", false, "none", "none"); }

// Test whether scores during decoding are the same as training
BOOST_AUTO_TEST_CASE(TestBeamDecodingScores) {
  shared_ptr<dynet::Model> mod;
  EncoderAttentionalPtr encatt;
  shared_ptr<EnsembleDecoder> ensdec;
  CreateModel(mod, encatt, ensdec);
  // Train
  float train_ll = 0, decode_ll = 0;
  Sentence decode_sent;
  // Perform decoding
  {
    ensdec->SetBeamSize(5);
    vector<EnsembleDecoderHypPtr> hyps = ensdec->GenerateNbest(sent_src_, 1);
    ensdec->SetBeamSize(1);
    decode_ll = hyps[0]->GetScore();
    decode_sent = hyps[0]->GetSentence();
  }
  // Calculate the training likelihood for that value
  {
    LLStats train_stat(vocab_trg_->size());
    dynet::ComputationGraph cg;
    encatt->NewGraph(cg);
    dynet::expr::Expression loss_expr = encatt->BuildSentGraph(sent_src_, decode_sent, cache_, nullptr, 0.f, false, cg, train_stat);
    train_ll = -as_scalar(cg.incremental_forward(loss_expr));
  }
  BOOST_CHECK_CLOSE(train_ll, decode_ll, 0.01);
}

// Test whether scores improve through beam search
BOOST_AUTO_TEST_CASE(TestBeamSearchImproves) {
  shared_ptr<dynet::Model> mod;
  EncoderAttentionalPtr encatt;
  shared_ptr<EnsembleDecoder> ensdec;
  CreateModel(mod, encatt, ensdec);
  // Beam
  size_t max_beam_size = 10;
  vector<float> scores(max_beam_size);
  for(size_t i = 0; i < max_beam_size; i++) {
    ensdec->SetBeamSize(i+1);
    vector<EnsembleDecoderHypPtr> hyps = ensdec->GenerateNbest(sent_src_, 1);
    scores[i] = hyps[0]->GetScore();
    if(i != 0) BOOST_CHECK_LE(scores[i-1], scores[i]);
  }
  ensdec->SetBeamSize(1);
}


BOOST_AUTO_TEST_SUITE_END()
