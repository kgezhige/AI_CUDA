#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor tensor = torch::rand({2, 3}).cuda();
    std::cout << tensor << std::endl;
    
    return 0;
}