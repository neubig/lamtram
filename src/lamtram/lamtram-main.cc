
#include <lamtram/lamtram.h>
#include <cnn/init.h>

using namespace lamtram;

int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);
    Cnntrans lamtram;
    return lamtram.main(argc, argv);
}
