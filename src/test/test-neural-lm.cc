#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <lamtram/macros.h>
#include <lamtram/neural-lm.h>
#include <lamtram/encoder-decoder.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/ensemble-decoder.h>
#include <lamtram/model-utils.h>
#include <dynet/dict.h>

using namespace std;
using namespace lamtram;

// ****** The fixture *******
struct TestNeuralLM {

  TestNeuralLM() : sent_src_(4), sent_trg_(4), cache_() {
    sent_src_ = {1, 2, 3, 0};
    sent_trg_ = {3, 2, 1, 0};
  }
  ~TestNeuralLM() { }

  Sentence sent_src_, sent_trg_, cache_;
};

// ****** The tests *******
BOOST_FIXTURE_TEST_SUITE(neural_lm, TestNeuralLM)

// Test whether reading and writing works.
// Note that this is just checking if serialized strings is equal,
// which is a hack for now because dynet::Model doesn't have an equality operator.
BOOST_AUTO_TEST_CASE(TestWriteRead) {
  std::shared_ptr<dynet::Model> act_mod(new dynet::Model), exp_mod(new dynet::Model);
  // Create a randomized lm
  DictPtr exp_vocab(CreateNewDict()); exp_vocab->convert("hello");
  NeuralLM exp_lm(exp_vocab, 2, 2, false, 3, BuilderSpec("rnn:2:1"), -1, "full", *exp_mod);
  // Write the LM
  ostringstream out;
  WriteDict(*exp_vocab, out);
  exp_lm.Write(out);
  ModelUtils::WriteModelText(out, *exp_mod);
  // Read the LM
  DictPtr act_src(new dynet::Dict), act_trg(new dynet::Dict);
  string first_string = out.str();
  istringstream in(out.str());
  NeuralLMPtr act_lm(ModelUtils::LoadMonolingualModel<NeuralLM>(in, act_mod, act_trg));
  // Write to a second string
  ostringstream out2;
  WriteDict(*act_trg, out2);
  act_lm->Write(out2);
  ModelUtils::WriteModelText(out2, *act_mod);
  string second_string = out2.str();
  // Check if the two
  BOOST_CHECK_EQUAL(first_string, second_string);
}

// Test whether scores during decoding are the same as those during training
BOOST_AUTO_TEST_CASE(TestDecodingScores) {
  std::shared_ptr<dynet::Model> mod(new dynet::Model);
  // Create a randomized lm
  DictPtr vocab(CreateNewDict()); vocab->convert("a"); vocab->convert("b"); vocab->convert("c");
  NeuralLMPtr lmptr(new NeuralLM(vocab, 1, 0, false, 3, BuilderSpec("rnn:2:1"), -1, "full", *mod));
  // Create the ensemble decoder
  vector<EncoderDecoderPtr> encdecs;
  vector<EncoderAttentionalPtr> encatts;
  vector<NeuralLMPtr> lms; lms.push_back(lmptr);
  EnsembleDecoder ensdec(encdecs, encatts, lms);
  // Compare the two values
  LLStats train_stat(vocab->size()), test_stat(vocab->size());
  vector<dynet::expr::Expression> layer_in;
  {
    dynet::ComputationGraph cg;
    lmptr->NewGraph(cg);
    dynet::expr::Expression loss_expr = lmptr->BuildSentGraph(sent_trg_, cache_, nullptr, nullptr, layer_in, 0.f, false, cg, train_stat);
    train_stat.loss_ += as_scalar(cg.incremental_forward(loss_expr));
  }
  vector<float> test_wordll;
  ensdec.CalcSentLL(sent_src_, sent_trg_, test_stat, test_wordll);
  BOOST_CHECK_CLOSE(train_stat.CalcPPL(), test_stat.CalcPPL(), 0.1);
}

BOOST_AUTO_TEST_SUITE_END()
