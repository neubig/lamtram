#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <lamtram/macros.h>
#include <lamtram/encoder-decoder.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/ensemble-decoder.h>
#include <lamtram/model-utils.h>
#include <dynet/training.h>
#include <dynet/dict.h>

using namespace std;
using namespace lamtram;

// ****** The fixture *******
struct TestEncoderDecoder {

  TestEncoderDecoder() : sent_src_(4), sent_trg_(4) {
    sent_src_ = {1, 2, 3, 0};
    sent_trg_ = {3, 2, 1, 0};

    // Create the model
    mod_ = shared_ptr<dynet::Model>(new dynet::Model);
    // Create a randomized lm
    vocab_src_ = DictPtr(CreateNewDict()); vocab_src_->convert("a"); vocab_src_->convert("b"); vocab_src_->convert("c");
    vocab_trg_ = DictPtr(CreateNewDict()); vocab_trg_->convert("x"); vocab_trg_->convert("y"); vocab_trg_->convert("z");
    NeuralLMPtr lmptr(new NeuralLM(vocab_trg_, 1, 0, false, 5, BuilderSpec("lstm:5:1"), -1, "full", *mod_));
    vector<LinearEncoderPtr> encs(1, LinearEncoderPtr(new LinearEncoder(vocab_src_->size(), 5, BuilderSpec("lstm:5:1"), -1, *mod_)));
    encdec_ = shared_ptr<EncoderDecoder>(new EncoderDecoder(encs, lmptr, *mod_));
    // Create the ensemble decoder
    vector<EncoderDecoderPtr> encdecs; encdecs.push_back(encdec_);
    vector<EncoderAttentionalPtr> encatts;
    vector<NeuralLMPtr> lms;
    ensdec_ = shared_ptr<EnsembleDecoder>(new EnsembleDecoder(encdecs, encatts, lms));
    ensdec_->SetSizeLimit(100);
    // Perform a few rounds of training
    dynet::SimpleSGDTrainer sgd(mod_.get());
    LLStats train_stat(vocab_trg_->size());
    for(size_t i = 0; i < 100; ++i) {
      dynet::ComputationGraph cg;
      encdec_->NewGraph(cg);
      dynet::expr::Expression loss_expr = encdec_->BuildSentGraph(sent_src_, sent_trg_, cache_, nullptr, 0.f, false, cg, train_stat);
      cg.forward(loss_expr);
      cg.backward(loss_expr);
      sgd.update(0.1);
    }
  }
  ~TestEncoderDecoder() { }

  Sentence sent_src_, sent_trg_, cache_;
  DictPtr vocab_src_, vocab_trg_;
  shared_ptr<EnsembleDecoder> ensdec_;
  EncoderDecoderPtr encdec_;
  shared_ptr<dynet::Model> mod_;
};

// ****** The tests *******
BOOST_FIXTURE_TEST_SUITE(encoder_decoder, TestEncoderDecoder)

// Test whether reading and writing works.
// Note that this is just checking if serialized strings is equal,
// which is a hack for now because dynet::Model doesn't have an equality operator.
BOOST_AUTO_TEST_CASE(TestWriteRead) {
  // Create a randomized lm
  shared_ptr<dynet::Model> act_mod(new dynet::Model), exp_mod(new dynet::Model);
  DictPtr exp_src_vocab(CreateNewDict()); exp_src_vocab->convert("hola");
  DictPtr exp_trg_vocab(CreateNewDict()); exp_trg_vocab->convert("hello");
  NeuralLMPtr exp_lm(new NeuralLM(exp_trg_vocab, 2, 2, false, 3, BuilderSpec("rnn:2:1"), -1, "full", *exp_mod));
  vector<LinearEncoderPtr> exp_encs(1, LinearEncoderPtr(new LinearEncoder(exp_src_vocab->size(), 2, BuilderSpec("rnn:2:1"), -1, *exp_mod)));
  EncoderDecoder exp_encatt(exp_encs, exp_lm, *exp_mod);
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

// Test whether scores during likelihood calculation are the same as training
BOOST_AUTO_TEST_CASE(TestLLScores) {
  // Compare the two values
  LLStats train_stat(vocab_trg_->size()), test_stat(vocab_trg_->size());
  {
    dynet::ComputationGraph cg;
    encdec_->NewGraph(cg);
    dynet::expr::Expression loss_expr = encdec_->BuildSentGraph(sent_src_, sent_trg_, cache_, nullptr, 0.f, false, cg, train_stat);
    train_stat.loss_ += as_scalar(cg.incremental_forward(loss_expr));
  }
  vector<float> test_wordll;
  ensdec_->CalcSentLL(sent_src_, sent_trg_, test_stat, test_wordll);
  BOOST_CHECK_CLOSE(train_stat.CalcPPL(), test_stat.CalcPPL(), 0.1);
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
    dynet::ComputationGraph cg;
    encdec_->NewGraph(cg);
    dynet::expr::Expression loss_expr = encdec_->BuildSentGraph(sent_src_, decode_sent, cache_, nullptr, 0.f, false, cg, train_stat);
    train_ll = -as_scalar(cg.incremental_forward(loss_expr));
  }
  BOOST_CHECK_CLOSE(train_ll, decode_ll, 0.01);
}

// Test whether scores during decoding are the same as training
BOOST_AUTO_TEST_CASE(TestBeamDecodingScores) {
  float train_ll = 0, decode_ll = 0;
  Sentence decode_sent;
  // Perform decoding
  {
    ensdec_->SetBeamSize(5);
    vector<EnsembleDecoderHypPtr> hyps = ensdec_->GenerateNbest(sent_src_, 1);
    ensdec_->SetBeamSize(1);
    decode_ll = hyps[0]->GetScore();
    decode_sent = hyps[0]->GetSentence();
  }
  // Calculate the training likelihood for that value
  {
    LLStats train_stat(vocab_trg_->size());
    dynet::ComputationGraph cg;
    encdec_->NewGraph(cg);
    dynet::expr::Expression loss_expr = encdec_->BuildSentGraph(sent_src_, decode_sent, cache_, nullptr, 0.f, false, cg, train_stat);
    train_ll = -as_scalar(cg.incremental_forward(loss_expr));
  }
  BOOST_CHECK_CLOSE(train_ll, decode_ll, 0.01);
}


BOOST_AUTO_TEST_SUITE_END()
