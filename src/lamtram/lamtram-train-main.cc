
#include <lamtram/lamtram-train.h>
#include <dynet/init.h>

using namespace lamtram;

int main(int argc, char** argv) {
    dynet::initialize(argc, argv);
    LamtramTrain train;
    return train.main(argc, argv);
}
