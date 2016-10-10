
#include <lamtram/lamtram.h>
#include <dynet/init.h>

using namespace lamtram;

int main(int argc, char** argv) {
    dynet::initialize(argc, argv);
    Lamtram lamtram;
    return lamtram.main(argc, argv);
}
