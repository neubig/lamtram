#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <lamtram/macros.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/model-utils.h>

using namespace std;
using namespace lamtram;

// ****** The fixture *******
struct TestEncoderAttentional {

    TestEncoderAttentional() : sent_(6) {
        sent_ = {0, 0, 1, 2, 3, 0};
    }
    ~TestEncoderAttentional() { }

    Sentence sent_;
};

// ****** The tests *******
BOOST_FIXTURE_TEST_SUITE(encoder_attentional, TestEncoderAttentional)

// Test whether reading and writing works.
// Note that this is just checking if serialized strings is equal,
// which is a hack for now because cnn::Model doesn't have an equality operator.
BOOST_AUTO_TEST_CASE(TestWriteRead) {
    // Create a randomized lm
    cnn::Model act_mod, exp_mod;
    cnn::VariableIndex empty_idx;
    NeuralLMPtr exp_lm(new NeuralLM(3, 2, 2, empty_idx, "rnn:2:1", -1, exp_mod));
    vector<LinearEncoderPtr> exp_encs(1, LinearEncoderPtr(new LinearEncoder(3, 2, "rnn:2:1", -1, exp_mod)));
    ExternAttentionalPtr exp_ext(new ExternAttentional(exp_encs, 2, 3, exp_mod));
    EncoderAttentional exp_encatt(exp_ext, exp_lm, exp_mod);
    // Write the LM
    ostringstream out;
    exp_encatt.Write(out);
    ModelUtils::WriteModelText(out, exp_mod);
    // Read the LM
    string first_string = out.str();
    istringstream in(out.str());
    EncoderAttentionalPtr act_lm(EncoderAttentional::Read(in, act_mod));
    ModelUtils::ReadModelText(in, act_mod);
    // Write to a second string
    ostringstream out2;
    act_lm->Write(out2);
    ModelUtils::WriteModelText(out2, act_mod);
    string second_string = out2.str();
    // Check if the two
    BOOST_CHECK_EQUAL(first_string, second_string);
}

BOOST_AUTO_TEST_SUITE_END()
