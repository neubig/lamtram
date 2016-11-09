
#include <lamtram/lamtram-train-multitask.h>
#include <dynet/init.h>

using namespace lamtram;

int main(int argc, char** argv) {
    dynet::initialize(argc, argv);
    LamtramTrainMultitask train;
    return train.main(argc, argv);
}
