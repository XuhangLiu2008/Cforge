//
// Created by xuhang liu on 2026/1/18.
//

#ifndef CFORGE_OPTIMIZER_H
#define CFORGE_OPTIMIZER_H

#include <iostream>
#include <torch/torch.h>
#include "Filaments.h"
#include "BatchExpectPassMatrix.h"
#include "_debugUtilis.h"
#include <map>

using namespace std;

class Optimizer { // abstract base class
    public:

        explicit Optimizer(FilaGroup* fila_group,
            const map<string, torch::Tensor> &configs,
            const torch::Tensor* target_pic_FRONTLIGHT,
            const torch::Tensor* target_pic_BACKLIGHT,
            const torch::Tensor* weight_FRONTLIGHT,
            const torch::Tensor* weight_BACKLIGHT) {

            this->fila_group = fila_group;
            this->configs = configs;

            this->target_pic_FRONTLIGHT = *target_pic_FRONTLIGHT;
            this->target_pic_BACKLIGHT = *target_pic_BACKLIGHT;
            this->weight_FRONTLIGHT = *weight_FRONTLIGHT;
            this->weight_BACKLIGHT = *weight_BACKLIGHT;
        };

        explicit Optimizer(FilaGroup* fila_group,
            const map<string, torch::Tensor> &configs,
            const torch::Tensor* target_pic_FRONTLIGHT,
            const torch::Tensor* target_pic_BACKLIGHT) {

            this->fila_group = fila_group;
            this->configs = configs;
            this->target_pic_FRONTLIGHT = *target_pic_FRONTLIGHT;
            this->target_pic_BACKLIGHT = *target_pic_BACKLIGHT;
        };

        FilaGroup* fila_group;

        torch::Tensor target_pic_FRONTLIGHT;
        torch::Tensor target_pic_BACKLIGHT;

        torch::Tensor weight_FRONTLIGHT;
        torch::Tensor weight_BACKLIGHT;

        map<string, torch::Tensor> configs;
        // depends

        virtual torch::Tensor solve();
        // BatchFilaLists returned

    private:

        unique_ptr<BatchExpectPassMatrix> core_mat_ptr;
};

// variations


// factory class


#endif //CFORGE_OPTIMIZER_H