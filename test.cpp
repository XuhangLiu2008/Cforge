#include <iostream>
#include <torch/torch.h>
#include "Filaments.h"
#include <filesystem>
#include "BatchExpectPassMatrix.h"
#include "_debugUtilis.h"

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

} // PASS

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
} // PASS

int _test_main_BatchExpectPassMatrix() {
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

    BatchExpectPassMatrix CoreCalc = BatchExpectPassMatrix(11, 12, &fila_group);

    torch::Tensor BatchFilaList = torch::zeros({11, 12}, torch::kUInt8);

    for (int i = 0;i < 11;i++) {
        for (int j = 1;j < i+1;j++) {
            BatchFilaList[i][j] = 1;
        }
    }

    cout << BatchFilaList << endl;

    CoreCalc.SetMatrix(&BatchFilaList);

    unique_ptr<torch::Tensor> outcome_front = CoreCalc.Solve_FRONTLIGHT();
    unique_ptr<torch::Tensor>outcome_back = CoreCalc.Solve_BACKLIGHT();

    // _info(*outcome_front);
    // _info(*outcome_back);

    // cout << outcome_front->cpu() << endl;
    // cout << outcome_back->cpu() << endl;

    cout << "FRONTLIGHT" << endl << BatchExpectPassMatrix::ExtractIntensity(outcome_front).second->to(torch::kCPU) << endl;
    cout << "BACKLIGHT" << endl << BatchExpectPassMatrix::ExtractIntensity(outcome_back).second->to(torch::kCPU) << endl;
}

int main() {
    std::cout << "current working dictionary: " << std::filesystem::current_path() << std::endl;
    cout << "START SUCCESSFULLY." << endl;
    _test_main_BatchExpectPassMatrix();

    return 0;
}