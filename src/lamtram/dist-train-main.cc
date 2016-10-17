
#include <lamtram/dist-train.h>
#include <dynet/init.h>

using namespace lamtram;

int main(int argc, char** argv) {
    DistTrain train;
    return train.main(argc, argv);
}
