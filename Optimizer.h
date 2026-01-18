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

using namespace std;

class Optimizer { // abstract base class
    public:
        Optimizer();
    private:
        BatchExpectPassMatrix core_mat;
};

// variaties


// factory class


#endif //CFORGE_OPTIMIZER_H