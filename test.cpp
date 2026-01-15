#include <iostream>
#include <torch/torch.h>
#include "Filaments.h"
#include <filesystem>
#include "BatchExpectPassMatrix.h"

using namespace std;

void print(const torch::Tensor &t) {
    cout << t << endl;
}

string currentPath() {
    return filesystem::path(__FILE__).parent_path().string();
}

torch::Tensor f() {
    return torch::tensor({1, 1, 1});
}

int _test_main_tmp() {
    cout << f() << endl;
    cout << 114514 << endl;
    return 0;

} // 你过关

int _test_main_FilaGroup() {

    cout << 114514 << endl;

    torch::Tensor refra_index = torch::tensor({9.47557073, 4.7540685, 4.45918725});
    torch::Tensor extinc_coeff = torch::tensor({0.23565427, 1.77511957, 6.07312633});

    Filament JAYO_orange = Filament("JAYO", "PLA_Basic_Orange",
        torch::tensor({255.0, 255.0, 0.0}, torch::kUInt8),
        refra_index, extinc_coeff);
    Filament AIR = Filament();

    vector<Filament> vec_fila;
    vec_fila.push_back(AIR);
    vec_fila.push_back(JAYO_orange);

    cout << torch::exp(torch::tensor({1, 1, 1})) << endl;

    FilaGroup fila_group = FilaGroup(2, 0.1, &vec_fila);

    fila_group.save(currentPath()+"/fila_group.txt");

    FilaGroup _fila_group = FilaGroup(currentPath()+"/fila_group.txt");

    _fila_group.save(currentPath()+"/_fila_group.txt");

    cout << 114514 << endl;

    return 0;
} // 你过关

int main() {
    std::cout << "current working dictionary: " << std::filesystem::current_path() << std::endl;
    cout << "START SUCCESSFULLY." << endl;
    _test_main_FilaGroup();

    return 0;
}