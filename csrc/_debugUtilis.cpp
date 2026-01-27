//
// Created by xuhang liu on 2026/1/18.
//

#include "_debugUtilis.h"

#include <iostream>
#include <torch/torch.h>
#include <memory>
#include <string>

using namespace std;

void _tensor_to_cout(const torch::Tensor* t, int precision) {
    if (t->dim() == 0) {
        cout << setprecision(precision) << t->item();
    }
    else {
        cout << "[ ";
        for (int i = 0;i < t->size(0);i++) {
            torch::Tensor t_tmp = (*t)[i];
            _tensor_to_cout(&t_tmp, precision);
            cout << " ";
        }
        cout << "]";
    }
}

void _info(const torch::Tensor t) {
    constexpr int precision = 4;

    cout << "device  : " << t.device() << endl;
    torch::Tensor t_cpu;
    if (t.device().type() == torch::kCPU) {t_cpu = t;}
    else t_cpu = t.cpu();

    cout << "sizes   : " << t_cpu.sizes() << endl;
    cout << "dtypes  : " << t_cpu.dtype() << endl;

    cout << "content : ";
    // expand the first dimension vertically
    torch::Tensor t_tmp = t_cpu[0];
    _tensor_to_cout(&t_tmp, precision);
    for (int i = 1;i < t_cpu.size(0);i++) {
        cout << endl << "          ";
        t_tmp = t_cpu[i];
        _tensor_to_cout(&t_tmp, precision);
    }
}
