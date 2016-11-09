#include "lamtram/rnn-cond.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>
#include <lamtram/macros.h>

#include "dynet/nodes.h"
#include "dynet/training.h"

using namespace std;
using namespace dynet;

namespace lamtram {

 dynet::expr::Expression RNNCONDBuilder::add_input_withContext( const Expression & x, Expression & attention_context,
        const dynet::expr::Expression & align_sum_in,
        bool train,
        dynet::ComputationGraph & cg,
        std::vector<dynet::expr::Expression> & align_out,
        dynet::expr::Expression & align_sum_out) {
        align_sum_in_ = &align_sum_in;
        align_out_ = &align_out;
        train_ = train;
        cg_ = &cg;
        align_sum_out_ = &align_sum_out;
        Expression out = add_input(x);
        attention_context = attention_context_;
        return out;
}



} // namespace dynet
