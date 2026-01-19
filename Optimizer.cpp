//
// Created by xuhang liu on 2026/1/18.
//

#include "Optimizer.h"

#include <random>
#include <torch/torch.h>



pair<unique_ptr<vector<int>>, unique_ptr<vector<int>>> simpleSimulatedAnnealing::_randDisturb() {
    // layer_index and target_fila
    // can be directly put input core_mat.BatchModify

    int lowerlimit = 1;
    int upperlimit = layer_size-2;

    // the data should be generated in the range [lowerlimit, upperlimit]
    // mean : (upperlimit-lowerlimit) / 2
    // default std_dev : (upperlimit-lowerlimit)/6

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist((upperlimit-lowerlimit) / 2, (upperlimit-lowerlimit)/6);

    vector<int> layer_index, target_fila;

    // random layer_index
    for (int _ = 0; _ < batch_size; _++) {
        int pos;

    }
}

torch::Tensor simpleSimulatedAnnealing::solve() {  // that is freaking dam sit rubbish

    batch_size = target_pic_FRONTLIGHT.numel(); // sizes = {H, W, 3}
    layer_size = configs["layer_size"];

    BatchExpectPassMatrix core_mat = BatchExpectPassMatrix(batch_size, layer_size, fila_group);
    this->core_mat_ptr = make_unique<BatchExpectPassMatrix>(core_mat);

    // Set initial values


    // SA
}