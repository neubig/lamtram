
#include <lamtram/lamtram-train.h>
#include <cnn/init.h>

using namespace lamtram;

int main(int argc, char** argv) {
    cnn::initialize(argc, argv);
    LamtramTrain train;
    return train.main(argc, argv);
}
