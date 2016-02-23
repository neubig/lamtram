#pragma once

#include <lamtram/dist-base.h>

namespace lamtram {

class DistFactory {

public:
  static DistPtr create_dist(const std::string & sig);
  static DistPtr from_file(const std::string & file_name, DictPtr dict);

};

}
