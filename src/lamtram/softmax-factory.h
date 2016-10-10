#pragma once

#include <lamtram/softmax-base.h>

namespace lamtram {

class SoftmaxFactory {

public:
  static SoftmaxPtr CreateSoftmax(const std::string & sig, int input_size, const DictPtr & vocab, dynet::Model & mod);

};

}
