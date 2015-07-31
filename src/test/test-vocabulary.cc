#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <lamtram/vocabulary.h>
#include <lamtram/macros.h>
#include <sstream>

using namespace std;
using namespace lamtram;

// ****** The tests *******
BOOST_AUTO_TEST_SUITE(vocabulary)

BOOST_AUTO_TEST_CASE(TestParseWords) {
    // Create the words. Sentence end symbol is always 0.
    Vocabulary vocab;
    string in = "  a b  c a c <s> b  ";
    Sentence exp(7), act;
    exp[0] = 1;
    exp[1] = 2;
    exp[2] = 3;
    exp[3] = 1;
    exp[4] = 3;
    exp[5] = 0;
    exp[6] = 2;
    act = vocab.ParseWords(in);
    BOOST_CHECK_EQUAL_COLLECTIONS(exp.begin(), exp.end(), act.begin(), act.end());
}

BOOST_AUTO_TEST_CASE(TestPrintWords) {
    Vocabulary vocab;
    string in = "  a b  c a c <s> b  ";
    string exp = "a b c a c <s> b";
    string act = vocab.PrintWords(vocab.ParseWords(in));
    BOOST_CHECK_EQUAL(exp, act);
}

BOOST_AUTO_TEST_CASE(TestReadWrite) {
    Vocabulary vocab_exp;
    string in = "  a b  c a c <s> b  ";
    vocab_exp.ParseWords(in);
    stringstream out;
    vocab_exp.Write(out);
    shared_ptr<Vocabulary> vocab_act(Vocabulary::Read(out));
    BOOST_CHECK_EQUAL_COLLECTIONS(
        vocab_exp.GetWIDs().begin(), vocab_exp.GetWIDs().end(),
        vocab_act->GetWIDs().begin(), vocab_act->GetWIDs().end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        vocab_exp.GetWSyms().begin(), vocab_exp.GetWSyms().end(),
        vocab_act->GetWSyms().begin(), vocab_act->GetWSyms().end());
}


BOOST_AUTO_TEST_SUITE_END()
