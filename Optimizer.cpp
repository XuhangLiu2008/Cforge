//
// Created by xuhang liu on 2026/1/18.
//

#include "Optimizer.h"

#include <random>
#include <torch/torch.h>



static random_device rd;
static mt19937 gen(rd());



pair<unique_ptr<vector<int>>, unique_ptr<vector<int>>> simpleSimulatedAnnealing::_randDisturb() {
    // layer_index and target_fila
    // can be directly put input core_mat.BatchModify

    vector<int> layer_index, target_fila;

    

    int lowerlimit = 1;
    int upperlimit = layer_size-2;

    // the data should be generated in the range [lowerlimit, upperlimit]
    // mean : (upperlimit-lowerlimit) / 2
    // default std_dev : (upperlimit-lowerlimit)/6

    float std_dev =  (upperlimit-lowerlimit)/6;
    if (configs.contains("std_dev"))
        std_dev = configs["std_dev"];
    normal_distribution<> norm_dist((upperlimit-lowerlimit) / 2, std_dev);

    // random layer_index
    for (int batch_index = 0; batch_index < batch_size; batch_index++) {
        int pos = static_cast<int>(norm_dist(gen));
        while (!((pos >= lowerlimit) &&
                 (pos <= upperlimit) &&
                 (
                     (core_mat_ptr->fila_list[batch_index][pos].item<int>() != 0)   ||
                     (core_mat_ptr->fila_list[batch_index][pos-1].item<int>() != 0) ||
                     (core_mat_ptr->fila_list[batch_index][pos+1].item<int>() != 0)
                 )
                 )) {
            pos = static_cast<int>(norm_dist(gen));
        }
        layer_index.push_back(pos);
    }



    uniform_real_distribution<> uni_float_dist(0, 1);
    uniform_int_distribution<> uni_int_dist(1, fila_group->num_fila-1);

    float air_ratio = configs.contains("air_ratio") ? configs["air_ratio"].get<float>() : 0.2;

    // random target_fila
    for (int batch_index = 0; batch_index < batch_size; batch_index++) {
        int pos = layer_index[batch_index];

        if (core_mat_ptr->fila_list[batch_index][pos].item<int>() == 0) { // extend
            target_fila.push_back(uni_int_dist(gen));
        }
        else if (core_mat_ptr->fila_list[batch_index][pos-1].item<int>() == 0 &&
                core_mat_ptr->fila_list[batch_index][pos+1].item<int>() == 0) {
                target_fila.push_back(uni_int_dist(gen)); // the only one in the list
        }
        else {
            target_fila.push_back(uni_float_dist(gen) < air_ratio ? 0 : uni_int_dist(gen));
            // remove or modify
        }

    }

    return make_pair(make_unique<vector<int>>(layer_index), make_unique<vector<int>>(target_fila));

}

torch::Tensor simpleSimulatedAnnealing::solve() {  // that is freaking dam sit rubbish

    batch_size = target_pic_FRONTLIGHT.numel() / 3; // sizes = {H, W, 3}
    layer_size = configs["layer_size"];

    BatchExpectPassMatrix core_mat = BatchExpectPassMatrix(batch_size, layer_size, fila_group);
    this->core_mat_ptr = make_unique<BatchExpectPassMatrix>(core_mat);

    // Set initial values

    uniform_int_distribution<> dist(1, fila_group->num_fila-1);
    // -1, except for AIR

    torch::Tensor BatchFilaList = torch::zeros({batch_size, layer_size}, torch::kUInt8);
    for (int batch_index = 0 ; batch_index < batch_size; batch_index++) {
        // init in the middle
        BatchFilaList[batch_index][(layer_size - 1) / 2] = dist(gen);
    }

    core_mat_ptr->SetMatrix(&BatchFilaList);

    // SA
}