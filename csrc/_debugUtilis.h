//
// Created by xuhang liu on 2026/1/18.
//

#ifndef CFORGE__DEBUGUTILIS_H
#define CFORGE__DEBUGUTILIS_H

#include <torch/torch.h>
#include <iostream>
#include <string>
#include <memory>

using namespace std;

void _tensor_to_cout(const torch::Tensor* t, int precision);

void _info(torch::Tensor t);

#endif //CFORGE__DEBUGUTILIS_H