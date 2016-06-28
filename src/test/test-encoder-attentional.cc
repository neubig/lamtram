#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <lamtram/macros.h>
#include <lamtram/encoder-decoder.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/ensemble-decoder.h>
#include <lamtram/model-utils.h>
#include <cnn/dict.h>
#include <cnn/training.h>

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
    vocab_src_ = DictPtr(CreateNewDict()); vocab_src_->Convert("a"); vocab_src_->Convert("b"); vocab_src_->Convert("c");
    vocab_trg_ = DictPtr(CreateNewDict()); vocab_trg_->Convert("x"); vocab_trg_->Convert("y"); vocab_trg_->Convert("z");

  }
  ~TestEncoderAttentional() { }

  void CreateModel(
        shared_ptr<cnn::Model> & mod,
        EncoderAttentionalPtr & encatt,
        shared_ptr<EnsembleDecoder> & ensdec,
        const std::string & attention_type = "mlp:2",
        bool attention_feed = false,
        const std::string & attention_hist = "none"
  ) {
    // Create the model
    mod = shared_ptr<cnn::Model>(new cnn::Model);
    NeuralLMPtr lmptr(new NeuralLM(vocab_trg_, 1, (attention_feed ? 5 : 0), attention_feed, 5, BuilderSpec("lstm:5:1"), -1, "full", *mod));
    vector<LinearEncoderPtr> encs(1, LinearEncoderPtr(new LinearEncoder(vocab_src_->size(), 5, BuilderSpec("lstm:5:1"), -1, *mod)));
    ExternAttentionalPtr ext(new ExternAttentional(encs, attention_type, attention_hist, 5, *mod));
    encatt = shared_ptr<EncoderAttentional>(new EncoderAttentional(ext, lmptr, *mod));
    // Create the ensemble decoder
    vector<EncoderDecoderPtr> encdecs;
    vector<EncoderAttentionalPtr> encatts; encatts.push_back(encatt);
    vector<NeuralLMPtr> lms;
    ensdec = shared_ptr<EnsembleDecoder>(new EnsembleDecoder(encdecs, encatts, lms));
    ensdec->SetSizeLimit(100);
    // Perform a few rounds of training
    cnn::SimpleSGDTrainer sgd(mod.get());
    LLStats train_stat(vocab_trg_->size());
    for(size_t i = 0; i < 100; ++i) {
      cnn::ComputationGraph cg;
      encatt->NewGraph(cg);
      encatt->BuildSentGraph(sent_src_, sent_trg_, cache_, 0.f, false, cg, train_stat);
      cg.forward();
      cg.backward();
      sgd.update(0.1);
    }
  }

  void TestDecoding(const std::string & attention_type, bool attention_feed, const std::string & attention_hist) {
    shared_ptr<cnn::Model> mod;
    EncoderAttentionalPtr encatt;
    shared_ptr<EnsembleDecoder> ensdec;
    CreateModel(mod, encatt, ensdec, attention_type, attention_feed, attention_hist);
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
      cnn::ComputationGraph cg;
      encatt->NewGraph(cg);
      encatt->BuildSentGraph(sent_src_, decode_sent, cache_, 0.f, false, cg, train_stat);
      train_ll = -as_scalar(cg.incremental_forward());
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
// which is a hack for now because cnn::Model doesn't have an equality operator.
BOOST_AUTO_TEST_CASE(TestWriteRead) {
  // Create a randomized lm
  shared_ptr<cnn::Model> act_mod(new cnn::Model), exp_mod(new cnn::Model);
  DictPtr exp_src_vocab(CreateNewDict()); exp_src_vocab->Convert("hola");
  DictPtr exp_trg_vocab(CreateNewDict()); exp_trg_vocab->Convert("hello");
  NeuralLMPtr exp_lm(new NeuralLM(exp_trg_vocab, 2, 2, false, 3, BuilderSpec("rnn:2:1"), -1, "full", *exp_mod));
  vector<LinearEncoderPtr> exp_encs(1, LinearEncoderPtr(new LinearEncoder(3, 2, BuilderSpec("rnn:2:1"), -1, *exp_mod)));
  ExternAttentionalPtr exp_ext(new ExternAttentional(exp_encs, "mlp:2", "none", 3, *exp_mod));
  EncoderAttentional exp_encatt(exp_ext, exp_lm, *exp_mod);
  // Write the Model
  ostringstream out;
  WriteDict(*exp_src_vocab, out);
  WriteDict(*exp_trg_vocab, out);
  exp_encatt.Write(out);
  ModelUtils::WriteModelText(out, *exp_mod);
  // Read the Model
  DictPtr act_src_vocab(new cnn::Dict), act_trg_vocab(new cnn::Dict);
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
BOOST_AUTO_TEST_CASE(TestLLScores) {
  shared_ptr<cnn::Model> mod;
  EncoderAttentionalPtr encatt;
  shared_ptr<EnsembleDecoder> ensdec;
  CreateModel(mod, encatt, ensdec);
  // Compare the two values
  LLStats train_stat(vocab_trg_->size()), test_stat(vocab_trg_->size());
  {
    cnn::ComputationGraph cg;
    encatt->NewGraph(cg);
    encatt->BuildSentGraph(sent_src_, sent_trg_, cache_, 0.f, false, cg, train_stat);
    train_stat.loss_ += as_scalar(cg.incremental_forward());
  }
  ensdec->CalcSentLL(sent_src_, sent_trg_, test_stat);
  BOOST_CHECK_CLOSE(train_stat.CalcPPL(), test_stat.CalcPPL(), 0.01);
}


// Test whether log likelihood is the same when batched or not
BOOST_AUTO_TEST_CASE(TestLLBatchScores) {
  shared_ptr<cnn::Model> mod;
  EncoderAttentionalPtr encatt;
  shared_ptr<EnsembleDecoder> ensdec;
  CreateModel(mod, encatt, ensdec, "mlp:5", true, "sum");
  LLStats batch_stat(vocab_trg_->size()), unbatch_stat(vocab_trg_->size());
  // Do unbatched calculation
  {
    cnn::ComputationGraph cg; encatt->NewGraph(cg);
    encatt->BuildSentGraph(sent_src_, sent_trg_, cache_, 0.f, false, cg, unbatch_stat);
    unbatch_stat.loss_ += as_scalar(cg.incremental_forward());
  }
  {
    cnn::ComputationGraph cg; encatt->NewGraph(cg);
    encatt->BuildSentGraph(sent_src2_, sent_trg2_, cache_, 0.f, false, cg, unbatch_stat);
    unbatch_stat.loss_ += as_scalar(cg.incremental_forward());
  }
  // Do batched calculation
  {
    std::vector<Sentence> batch_src(2); batch_src[0] = sent_src_; batch_src[1] = sent_src2_;
    std::vector<Sentence> batch_trg(2); batch_trg[0] = sent_trg_; batch_trg[1] = sent_trg2_;
    std::vector<Sentence> batch_cache(2); batch_cache[0] = cache_; batch_cache[1] = cache_;
    cnn::ComputationGraph cg; encatt->NewGraph(cg);
    encatt->BuildSentGraph(batch_src, batch_trg, batch_cache, 0.f, false, cg, batch_stat);
    batch_stat.loss_ += as_scalar(cg.incremental_forward());
  }
  BOOST_CHECK_CLOSE(unbatch_stat.CalcPPL(), batch_stat.CalcPPL(), 0.5);
}

// Test whether scores during decoding are the same as training
BOOST_AUTO_TEST_CASE(TestDecodingDotFalseNone)   { TestDecoding("dot",   false, "none"); }
BOOST_AUTO_TEST_CASE(TestDecodingDotTrueNone)    { TestDecoding("dot",   true,  "none"); }
BOOST_AUTO_TEST_CASE(TestDecodingDotFalseSum)    { TestDecoding("dot",   false, "sum" ); }
BOOST_AUTO_TEST_CASE(TestDecodingMLPFalseNone)   { TestDecoding("mlp:5", false, "none"); }
BOOST_AUTO_TEST_CASE(TestDecodingMLPTrueSum)     { TestDecoding("mlp:5", true,  "sum" ); }
BOOST_AUTO_TEST_CASE(TestDecodingBilinFalseNone) { TestDecoding("bilin", false, "none"); }

// Test whether scores during decoding are the same as training
BOOST_AUTO_TEST_CASE(TestBeamDecodingScores) {
  shared_ptr<cnn::Model> mod;
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
    cnn::ComputationGraph cg;
    encatt->NewGraph(cg);
    encatt->BuildSentGraph(sent_src_, decode_sent, cache_, 0.f, false, cg, train_stat);
    train_ll = -as_scalar(cg.incremental_forward());
  }
  BOOST_CHECK_CLOSE(train_ll, decode_ll, 0.01);
}

// Test whether scores improve through beam search
BOOST_AUTO_TEST_CASE(TestBeamSearchImproves) {
  shared_ptr<cnn::Model> mod;
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
