#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <lamtram/macros.h>
#include <lamtram/neural-lm.h>
#include <lamtram/model-utils.h>
#include <cnn/dict.h>

using namespace std;
using namespace lamtram;

// ****** The fixture *******
struct TestNeuralLM {

  TestNeuralLM() : sent_(6) {
    sent_ = {0, 0, 1, 2, 3, 0};
  }
  ~TestNeuralLM() { }

  Sentence sent_;
};

// ****** The tests *******
BOOST_FIXTURE_TEST_SUITE(neural_lm, TestNeuralLM)

// Test whether reading and writing works.
// Note that this is just checking if serialized strings is equal,
// which is a hack for now because cnn::Model doesn't have an equality operator.
BOOST_AUTO_TEST_CASE(TestWriteRead) {
  std::shared_ptr<cnn::Model> act_mod(new cnn::Model), exp_mod(new cnn::Model);
  // Create a randomized lm
  DictPtr exp_vocab(CreateNewDict()); exp_vocab->Convert("hello");
  cnn::VariableIndex empty_idx;
  NeuralLM exp_lm(exp_vocab, 2, 2, false, empty_idx, BuilderSpec("rnn:2:1"), -1, "full", *exp_mod);
  // Write the LM
  ostringstream out;
  WriteDict(*exp_vocab, out);
  exp_lm.Write(out);
  ModelUtils::WriteModelText(out, *exp_mod);
  // Read the LM
  DictPtr act_src(new cnn::Dict), act_trg(new cnn::Dict);
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

BOOST_AUTO_TEST_SUITE_END()
