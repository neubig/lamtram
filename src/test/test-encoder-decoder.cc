#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <lamtram/macros.h>
#include <lamtram/encoder-decoder.h>
#include <lamtram/model-utils.h>
#include <cnn/dict.h>

using namespace std;
using namespace lamtram;

// ****** The fixture *******
struct TestEncoderDecoder {

  TestEncoderDecoder() : sent_(6) {
    sent_ = {0, 0, 1, 2, 3, 0};
  }
  ~TestEncoderDecoder() { }

  Sentence sent_;
};

// ****** The tests *******
BOOST_FIXTURE_TEST_SUITE(encoder_decoder, TestEncoderDecoder)

// Test whether reading and writing works.
// Note that this is just checking if serialized strings is equal,
// which is a hack for now because cnn::Model doesn't have an equality operator.
BOOST_AUTO_TEST_CASE(TestWriteRead) {
  // Create a randomized lm
  shared_ptr<cnn::Model> act_mod(new cnn::Model), exp_mod(new cnn::Model);
  cnn::VariableIndex empty_idx;
  DictPtr exp_src_vocab(new cnn::Dict); exp_src_vocab->Convert("<s>"); exp_src_vocab->Convert("<unk>"); exp_src_vocab->Convert("hola");
  DictPtr exp_trg_vocab(new cnn::Dict); exp_trg_vocab->Convert("<s>"); exp_trg_vocab->Convert("<unk>"); exp_trg_vocab->Convert("hello");
  NeuralLMPtr exp_lm(new NeuralLM(exp_trg_vocab, 2, 2, false, empty_idx, BuilderSpec("rnn:2:1"), -1, "full", *exp_mod));
  vector<LinearEncoderPtr> exp_encs(1, LinearEncoderPtr(new LinearEncoder(3, 2, BuilderSpec("rnn:2:1"), -1, *exp_mod)));
  EncoderDecoder exp_encatt(exp_encs, exp_lm, *exp_mod);
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
  EncoderDecoderPtr act_lm(ModelUtils::LoadBilingualModel<EncoderDecoder>(in, act_mod, act_src_vocab, act_trg_vocab));
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

BOOST_AUTO_TEST_SUITE_END()
