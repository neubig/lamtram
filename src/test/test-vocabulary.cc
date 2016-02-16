#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <lamtram/macros.h>
#include <lamtram/sentence.h>
#include <lamtram/dict-utils.h>
#include <cnn/dict.h>
#include <sstream>

using namespace std;
using namespace lamtram;

// ****** The tests *******
BOOST_AUTO_TEST_SUITE(vocabulary)

BOOST_AUTO_TEST_CASE(TestParseWords) {
    // Create the words. Sentence end symbol is always 0.
    cnn::Dict vocab;
    string in = "  a b  c a c <s> b  ";
    Sentence exp(7), act;
    exp[0] = 1;
    exp[1] = 2;
    exp[2] = 3;
    exp[3] = 1;
    exp[4] = 3;
    exp[5] = 0;
    exp[6] = 2;
    act = ParseWords(vocab, in, false);
    BOOST_CHECK_EQUAL_COLLECTIONS(exp.begin(), exp.end(), act.begin(), act.end());
}

BOOST_AUTO_TEST_CASE(TestPrintWords) {
    cnn::Dict vocab;
    string in = "  a b  c a c <s> b  ";
    string exp = "a b c a c <s> b";
    string act = PrintWords(vocab, ParseWords(vocab, in, false));
    BOOST_CHECK_EQUAL(exp, act);
}

BOOST_AUTO_TEST_CASE(TestReadWrite) {
    cnn::Dict vocab_exp;
    string in = "  a b  c a c <s> b  ";
    ParseWords(vocab_exp, in, false);
    stringstream out;
    WriteDict(vocab_exp, out);
    shared_ptr<cnn::Dict> vocab_act(ReadDict(out));
    BOOST_CHECK_EQUAL_COLLECTIONS(
        vocab_exp.GetWords().begin(), vocab_exp.GetWords().end(),
        vocab_act->GetWords().begin(), vocab_act->GetWords().end());
}


BOOST_AUTO_TEST_SUITE_END()
