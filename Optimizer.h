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
#include <utility>
#include "json.hpp"

using namespace std;

class Optimizer { // abstract base class
    public:

        explicit Optimizer(FilaGroup* fila_group,
            const nlohmann::json &configs,
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
            const nlohmann::json &configs,
            const torch::Tensor* target_pic_FRONTLIGHT,
            const torch::Tensor* target_pic_BACKLIGHT) {

            this->fila_group = fila_group;
            this->configs = configs;

            this->target_pic_FRONTLIGHT = *target_pic_FRONTLIGHT;
            this->target_pic_BACKLIGHT = *target_pic_BACKLIGHT;
            this->weight_FRONTLIGHT = torch::ones({this->target_pic_FRONTLIGHT.size(0), this->target_pic_FRONTLIGHT.size(1)});
            this->weight_BACKLIGHT = torch::ones({this->target_pic_BACKLIGHT.size(0), this->target_pic_BACKLIGHT.size(1)});
        };

        FilaGroup* fila_group;

        torch::Tensor target_pic_FRONTLIGHT; // sizes = {H, W, 3}
        torch::Tensor target_pic_BACKLIGHT; // sizes = {H, W, 3}

        torch::Tensor weight_FRONTLIGHT; // sizes = {H, W}
        torch::Tensor weight_BACKLIGHT; // sizes = {H, W}

        nlohmann::json configs;
        // depends

        virtual torch::Tensor solve();
        // BatchFilaLists returned

    protected:

        virtual pair<
        pair<unique_ptr<vector<int>>, unique_ptr<vector<int>>>,
        pair<unique_ptr<vector<int>>, unique_ptr<vector<int>>> >
        _randDisturb();

        virtual void _checkConfigs();

        unique_ptr<BatchExpectPassMatrix> core_mat_ptr;

        pair<unique_ptr<torch::Tensor>, unique_ptr<torch::Tensor>> _solveMat() {
            pair<unique_ptr<torch::Tensor>, unique_ptr<torch::Tensor> > FRONTLIGHT_intensity_pair =
                    BatchExpectPassMatrix::ExtractIntensity(this->core_mat_ptr->Solve_FRONTLIGHT());

            pair<unique_ptr<torch::Tensor>, unique_ptr<torch::Tensor> > BACKLIGHT_intensity_pair =
                    BatchExpectPassMatrix::ExtractIntensity(this->core_mat_ptr->Solve_BACKLIGHT());

            return make_pair(std::move(FRONTLIGHT_intensity_pair.second), std::move(BACKLIGHT_intensity_pair.second));
        }
};

// variations

class simpleSimulatedAnnealing : public Optimizer{ // that is freaking dam sit rubbish

    pair<
    pair<unique_ptr<vector<int>>, unique_ptr<vector<int>>>,
    pair<unique_ptr<vector<int>>, unique_ptr<vector<int>>> >
    _randDisturb() override;

    torch::Tensor solve() override;

    private:

        unique_ptr<torch::Tensor> _loss();

        int batch_size;
        int layer_size;

        torch::Tensor t_f;
        torch::Tensor t_b;
        torch::Tensor w_ft;
        torch::Tensor w_bt;

    /*
     config: {
        layer_size : int,
        std_dev : float?,
        air_ratio : float?,
        base_extinc_coeff : float,
        rgb_weight : {float, float, float}?,
        sa_params : {
            init_temperature : float,
            min_temperature : float,
            cooling_rate : float
        }
     }
     */
};


// factory class


#endif //CFORGE_OPTIMIZER_H