#pragma once

#include <string>

namespace lamtram {

class DistTrain {

public:
  DistTrain() { }

  int main(int argc, char** argv);
  
protected:

  // Variable settings
  std::string model_out_file_;
  std::string train_file_;

};

}
