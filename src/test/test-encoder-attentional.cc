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

    // Create the model
    mod_ = shared_ptr<cnn::Model>(new cnn::Model);
    // Create a randomized lm
    vocab_src_ = DictPtr(CreateNewDict()); vocab_src_->Convert("a"); vocab_src_->Convert("b"); vocab_src_->Convert("c");
    vocab_trg_ = DictPtr(CreateNewDict()); vocab_trg_->Convert("x"); vocab_trg_->Convert("y"); vocab_trg_->Convert("z");
    NeuralLMPtr lmptr(new NeuralLM(vocab_trg_, 1, 0, false, 5, BuilderSpec("lstm:5:1"), -1, "full", *mod_));
    vector<LinearEncoderPtr> encs(1, LinearEncoderPtr(new LinearEncoder(vocab_src_->size(), 5, BuilderSpec("lstm:5:1"), -1, *mod_)));
    ExternAttentionalPtr ext(new ExternAttentional(encs, "mlp:2", "none", 3, *mod_));
    encatt_ = shared_ptr<EncoderAttentional>(new EncoderAttentional(ext, lmptr, *mod_));
    // Create the ensemble decoder
    vector<EncoderDecoderPtr> encdecs;
    vector<EncoderAttentionalPtr> encatts; encatts.push_back(encatt_);
    vector<NeuralLMPtr> lms;
    ensdec_ = shared_ptr<EnsembleDecoder>(new EnsembleDecoder(encdecs, encatts, lms));
    // Perform a few rounds of training
    cnn::SimpleSGDTrainer sgd(mod_.get());
    LLStats train_stat(vocab_trg_->size());
    for(size_t i = 0; i < 100; ++i) {
      cnn::ComputationGraph cg;
      encatt_->NewGraph(cg);
      encatt_->BuildSentGraph(sent_src_, sent_trg_, cache_, 0.f, false, cg, train_stat);
      cg.forward();
      cg.backward();
      sgd.update(0.1);
    }
  }
  ~TestEncoderAttentional() { }

  Sentence sent_src_, sent_trg_, cache_;
  shared_ptr<EnsembleDecoder> ensdec_;
  DictPtr vocab_src_, vocab_trg_;
  EncoderAttentionalPtr encatt_;
  std::shared_ptr<cnn::Model> mod_;
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
  // Compare the two values
  LLStats train_stat(vocab_trg_->size()), test_stat(vocab_trg_->size());
  {
    cnn::ComputationGraph cg;
    encatt_->NewGraph(cg);
    encatt_->BuildSentGraph(sent_src_, sent_trg_, cache_, 0.f, false, cg, train_stat);
    train_stat.loss_ += as_scalar(cg.incremental_forward());
  }
  ensdec_->CalcSentLL(sent_src_, sent_trg_, test_stat);
  BOOST_CHECK_EQUAL(train_stat.CalcPPL(), test_stat.CalcPPL());
}

// Test whether scores during decoding are the same as training
BOOST_AUTO_TEST_CASE(TestDecodingScores) {
  float train_ll = 0, decode_ll = 0;
  Sentence decode_sent;
  // Perform decoding
  {
    vector<EnsembleDecoderHypPtr> hyps = ensdec_->GenerateNbest(sent_src_, 1);
    decode_ll = hyps[0]->GetScore();
    decode_sent = hyps[0]->GetSentence();
  }
  // Calculate the training likelihood for that value
  {
    LLStats train_stat(vocab_trg_->size());
    cnn::ComputationGraph cg;
    encatt_->NewGraph(cg);
    encatt_->BuildSentGraph(sent_src_, decode_sent, cache_, 0.f, false, cg, train_stat);
    train_ll = -as_scalar(cg.incremental_forward());
  }
  BOOST_CHECK_CLOSE(train_ll, decode_ll, 0.01);
}

BOOST_AUTO_TEST_SUITE_END()
