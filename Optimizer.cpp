//
// Created by xuhang liu on 2026/1/18.
//

#include "Optimizer.h"

#include <random>
#include <torch/torch.h>



static random_device rd;
static mt19937 gen(rd());



pair<
pair<unique_ptr<torch::Tensor>, unique_ptr<torch::Tensor>>,
pair<unique_ptr<torch::Tensor>, unique_ptr<torch::Tensor>> >
simpleSimulatedAnnealing::_randDisturb() {
    // layer_index and target_fila
    // can be directly put input core_mat.BatchModify

    // pair<
    // pair<*layer_index_forward, *target_fila_forward>,
    // pair<*layer_index_backward, *target_fila_backward>
    // >


    torch::Tensor layer_index = torch::zeros({batch_size}),
                  target_fila = torch::zeros({batch_size});
    torch::Tensor r_layer_index = torch::zeros({batch_size}),
                  r_target_fila = torch::zeros({batch_size}); // the reversed transformation that can reverse all the modification


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
        layer_index[batch_index] = pos;
        r_layer_index[batch_index] = pos;
    }



    uniform_real_distribution<> uni_float_dist(0, 1);
    uniform_int_distribution<> uni_int_dist(1, fila_group->num_fila-1);

    float air_ratio = configs.contains("air_ratio") ? configs["air_ratio"].get<float>() : 0.2;

    // random target_fila
    for (int batch_index = 0; batch_index < batch_size; batch_index++) {
        int pos = layer_index[batch_index].item<int>();

        if (core_mat_ptr->fila_list[batch_index][pos].item<int>() == 0) { // extend
            target_fila[batch_index] = uni_int_dist(gen); // random filament
            r_target_fila[batch_index] = core_mat_ptr->fila_list[batch_index][pos].item<int>(); // original filament
        }
        else if (core_mat_ptr->fila_list[batch_index][pos-1].item<int>() == 0 &&
                core_mat_ptr->fila_list[batch_index][pos+1].item<int>() == 0) {
            target_fila[batch_index] = uni_int_dist(gen); // random filament
            r_target_fila[batch_index] = core_mat_ptr->fila_list[batch_index][pos].item<int>(); // original filament
        }
        else {
            target_fila[batch_index] = uni_float_dist(gen) < air_ratio ? 0 : uni_int_dist(gen);
            // remove or modify
            r_target_fila[batch_index] = core_mat_ptr->fila_list[batch_index][pos].item<int>();
        }

    }

    return make_pair(
            make_pair(make_unique<torch::Tensor>(layer_index), make_unique<torch::Tensor>(target_fila)),
            make_pair(make_unique<torch::Tensor>(r_layer_index), make_unique<torch::Tensor>(r_target_fila))
        );

}

unique_ptr<torch::Tensor> simpleSimulatedAnnealing::_loss() {
    // ALL PARAMS:
    // loss([cr_f, cg_f, cb_f], [tr_f, tg_f, tb_f],
    //      [cr_b, cg_b, cb_b], [tr_b, tg_b, tb_b],
    //      [w_r, w_g, w_b]
    //      [w_ft, w_bt])


    // MSE (to be simple)
    // ----------------------------------------------------------------------------------------------------------------------
    // | item   | (cr_f - tr_f)^2 | (cg_f - tg_f)^2 | (cb_f - tb_f)^2 | (cr_b - tr_b)^2 | (cg_b - tg_b)^2 | (cb_b - tb_b)^2 |
    // | weight |   w_r * w_ft    |   w_g * w_ft    |   w_b * w_ft    |   w_r * w_bt    |   w_g * w_bt    |   w_b * w_bt    |
    // ----------------------------------------------------------------------------------------------------------------------


    auto tmp = _solveMat();
    torch::Tensor c_f = *tmp.first / configs["base_extinc_coeff"].get<float>(); // sizes : {H*W, 3}
    torch::Tensor c_b = *tmp.second; // sizes : {H*W, 3}

    torch::Tensor rgb_w = torch::tensor({1.0, 1.0, 1.0});
    if (configs.contains("rgb_weight"))
        rgb_w = torch::tensor({configs["rgb_weight"][0].get<float>(),
                                  configs["rgb_weight"][1].get<float>(),
                                  configs["rgb_weight"][2].get<float>()});

    torch::Tensor tot_w_f = torch::matmul(w_ft.t(), rgb_w);
    torch::Tensor tot_w_b = torch::matmul(w_bt.t(), rgb_w);

    return make_unique<torch::Tensor>(torch::sum(tot_w_f * torch::pow(c_f - t_f, 2) +
        tot_w_b * torch::pow(c_b - t_b, 2), 1));

}

torch::Tensor _metropolis_mask(float cur_temperature,
                               unique_ptr<torch::Tensor> pre_loss,
                               unique_ptr<torch::Tensor> cur_loss) {

    // the mask(dtype = bool) is used to be acted on the reversed modifier
    // so the mask is exactly the opposite of the normal criteria
    // true -> reject, false -> accept

    // metropolis:
    // accept if
    // E' < E or rand[0, 1) < exp(delta_E / T)
    return (*cur_loss > *pre_loss) * (torch::exp(-(*cur_loss - *pre_loss) / cur_temperature) < torch::rand_like(*pre_loss));
}

torch::Tensor simpleSimulatedAnnealing::solve() {  // that is freaking dam sit rubbish

    batch_size = target_pic_FRONTLIGHT.numel() / 3; // sizes = {H, W, 3}
    layer_size = configs["layer_size"];



    t_f = target_pic_FRONTLIGHT.to(torch::kMPS).flatten(0, 1); // {batch_size, 3}
    t_b = target_pic_BACKLIGHT.to(torch::kMPS).flatten(0, 1);  // {batch_size, 3}
    w_ft = weight_FRONTLIGHT.to(torch::kMPS).flatten(0, 1);    // {batch_size}
    w_bt = weight_BACKLIGHT.to(torch::kMPS).flatten(0, 1);     // {batch_size}



    // init core_mat

    BatchExpectPassMatrix core_mat = BatchExpectPassMatrix(batch_size, layer_size, fila_group);
    this->core_mat_ptr = make_unique<BatchExpectPassMatrix>(core_mat);



    // Set initial values

    uniform_int_distribution<> uni_int_dist(1, fila_group->num_fila-1);
    // -1, except for AIR

    torch::Tensor BatchFilaList = torch::zeros({batch_size, layer_size}, torch::kUInt8);
    for (int batch_index = 0 ; batch_index < batch_size; batch_index++) {
        // init in the middle
        BatchFilaList[batch_index][(layer_size - 1) / 2] = uni_int_dist(gen);
    }

    core_mat_ptr->SetMatrix(&BatchFilaList);



    // SA
    unique_ptr<torch::Tensor> pre_loss = _loss();
    unique_ptr<torch::Tensor> cur_loss = nullptr;

    float init_temperature = configs["init_temperature"].get<float>();
    float cooling_rate = configs["cooling_rate"].get<float>();
    float min_temperature = configs["min_temperature"].get<float>();

    float cur_temperature = init_temperature;

    while (cur_temperature > min_temperature) {

        auto bidirect_modifier = _randDisturb();

        auto modifier = std::move(bidirect_modifier.first);
        auto reversed_modifier = std::move(bidirect_modifier.second);

        core_mat_ptr -> BatchModify(std::move(modifier.first), std::move(modifier.second));

        cur_loss = _loss();

        torch::Tensor reversed_mask = _metropolis_mask(cur_temperature,
                                               std::move(pre_loss),
                                               std::move(cur_loss));

        torch::Tensor reversed_layer_index = *reversed_modifier.first,

        cur_temperature *= cooling_rate;
    }

}