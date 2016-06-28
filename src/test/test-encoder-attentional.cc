#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <lamtram/macros.h>
#include <lamtram/encoder-decoder.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/ensemble-decoder.h>
#include <lamtram/model-utils.h>
#include <cnn/dict.h>

using namespace std;
using namespace lamtram;

// ****** The fixture *******
struct TestEncoderAttentional {

  TestEncoderAttentional() : sent_src_(4), sent_trg_(4) {
    sent_src_ = {1, 2, 3, 0};
    sent_trg_ = {3, 2, 1, 0};
  }
  ~TestEncoderAttentional() { }

  Sentence sent_src_, sent_trg_, cache_;
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
  std::shared_ptr<cnn::Model> mod(new cnn::Model);
  // Create a randomized lm
  DictPtr vocab_src(CreateNewDict()); vocab_src->Convert("a"); vocab_src->Convert("b"); vocab_src->Convert("c");
  DictPtr vocab_trg(CreateNewDict()); vocab_trg->Convert("x"); vocab_trg->Convert("y"); vocab_trg->Convert("z");
  NeuralLMPtr lmptr(new NeuralLM(vocab_trg, 2, 2, true, 3, BuilderSpec("rnn:2:1"), -1, "full", *mod));
  vector<LinearEncoderPtr> encs(1, LinearEncoderPtr(new LinearEncoder(vocab_src->size(), 2, BuilderSpec("rnn:2:1"), -1, *mod)));
  ExternAttentionalPtr ext(new ExternAttentional(encs, "mlp:2", "sum", 2, *mod));
  EncoderAttentionalPtr encatt(new EncoderAttentional(ext, lmptr, *mod));
  // Create the ensemble decoder
  vector<EncoderDecoderPtr> encdecs;
  vector<EncoderAttentionalPtr> encatts; encatts.push_back(encatt);
  vector<NeuralLMPtr> lms;
  EnsembleDecoder ensdec(encdecs, encatts, lms);
  // Compare the two values
  LLStats train_stat(vocab_trg->size()), test_stat(vocab_trg->size());
  {
    cnn::ComputationGraph cg;
    encatt->NewGraph(cg);
    encatt->BuildSentGraph(sent_src_, sent_trg_, cache_, 0.f, false, cg, train_stat);
    train_stat.loss_ += as_scalar(cg.incremental_forward());
  }
  ensdec.CalcSentLL(sent_src_, sent_trg_, test_stat);
  BOOST_CHECK_EQUAL(train_stat.CalcPPL(), test_stat.CalcPPL());
}

// Test whether scores during likelihood calculation are the same as training
BOOST_AUTO_TEST_CASE(TestLLScoresDot) {
  std::shared_ptr<cnn::Model> mod(new cnn::Model);
  // Create a randomized lm
  DictPtr vocab_src(CreateNewDict()); vocab_src->Convert("a"); vocab_src->Convert("b"); vocab_src->Convert("c");
  DictPtr vocab_trg(CreateNewDict()); vocab_trg->Convert("x"); vocab_trg->Convert("y"); vocab_trg->Convert("z");
  NeuralLMPtr lmptr(new NeuralLM(vocab_trg, 2, 2, false, 3, BuilderSpec("rnn:2:1"), -1, "full", *mod));
  vector<LinearEncoderPtr> encs(1, LinearEncoderPtr(new LinearEncoder(vocab_src->size(), 2, BuilderSpec("rnn:2:1"), -1, *mod)));
  ExternAttentionalPtr ext(new ExternAttentional(encs, "dot", "none", 2, *mod));
  EncoderAttentionalPtr encatt(new EncoderAttentional(ext, lmptr, *mod));
  // Create the ensemble decoder
  vector<EncoderDecoderPtr> encdecs;
  vector<EncoderAttentionalPtr> encatts; encatts.push_back(encatt);
  vector<NeuralLMPtr> lms;
  EnsembleDecoder ensdec(encdecs, encatts, lms);
  // Compare the two values
  LLStats train_stat(vocab_trg->size()), test_stat(vocab_trg->size());
  {
    cnn::ComputationGraph cg;
    encatt->NewGraph(cg);
    encatt->BuildSentGraph(sent_src_, sent_trg_, cache_, 0.f, false, cg, train_stat);
    train_stat.loss_ += as_scalar(cg.incremental_forward());
  }
  ensdec.CalcSentLL(sent_src_, sent_trg_, test_stat);
  BOOST_CHECK_EQUAL(train_stat.CalcPPL(), test_stat.CalcPPL());
}


BOOST_AUTO_TEST_SUITE_END()
