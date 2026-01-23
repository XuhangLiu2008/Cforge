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
        virtual ~Optimizer() = default;

        explicit Optimizer(FilaGroup* fila_group,
                           const nlohmann::json &configs,
                           const torch::Tensor* target_pic_FRONTLIGHT,
                           const torch::Tensor* target_pic_BACKLIGHT,
                           const torch::Tensor* weight_FRONTLIGHT,
                           const torch::Tensor* weight_BACKLIGHT) {

            this->fila_group = fila_group;
            this->configs = configs;

            if (this->target_pic_BACKLIGHT.sizes() != this->target_pic_FRONTLIGHT.sizes() ||
                this->target_pic_FRONTLIGHT.sizes() != this->weight_FRONTLIGHT.sizes() ||
                this->weight_FRONTLIGHT.sizes() !=  this->weight_BACKLIGHT.sizes()) {
                throw std::length_error("Inconsistent tensor sizes");
            }

            this->target_pic_FRONTLIGHT = *target_pic_FRONTLIGHT;
            this->target_pic_BACKLIGHT = *target_pic_BACKLIGHT;
            this->weight_FRONTLIGHT = *weight_FRONTLIGHT;
            this->weight_BACKLIGHT = *weight_BACKLIGHT;

            _checkConfigs();
        };

        explicit Optimizer(FilaGroup* fila_group,
            const nlohmann::json &configs,
            const torch::Tensor* target_pic_FRONTLIGHT,
            const torch::Tensor* target_pic_BACKLIGHT) {

            this->fila_group = fila_group;
            this->configs = configs;

            if (this->target_pic_BACKLIGHT.sizes() != this->target_pic_FRONTLIGHT.sizes()) {
                throw std::length_error("Inconsistent tensor sizes");
            }

            this->target_pic_FRONTLIGHT = *target_pic_FRONTLIGHT;
            this->target_pic_BACKLIGHT = *target_pic_BACKLIGHT;
            this->weight_FRONTLIGHT = torch::ones_like(this->target_pic_FRONTLIGHT);
            this->weight_BACKLIGHT = torch::ones_like(this->target_pic_BACKLIGHT);

            _checkConfigs();
        };

        FilaGroup* fila_group;

        torch::Tensor target_pic_FRONTLIGHT; // sizes = {H, W, 3}
        torch::Tensor target_pic_BACKLIGHT; // sizes = {H, W, 3}

        torch::Tensor weight_FRONTLIGHT; // sizes = {H, W}
        torch::Tensor weight_BACKLIGHT; // sizes = {H, W}

        nlohmann::json configs;
        // depends

        virtual pair<pair<unique_ptr<torch::Tensor>, unique_ptr<torch::Tensor>> , unique_ptr<torch::Tensor>> solve();
        // <<FRONTLIGHT_intensity, BACKLIGHT_intensity> fila_lists> returned

    protected:

        virtual pair<
            pair<unique_ptr<torch::Tensor>, unique_ptr<torch::Tensor>>,
            pair<unique_ptr<torch::Tensor>, unique_ptr<torch::Tensor>> >
        _randDisturb();

        virtual void _checkConfigs();

        unique_ptr<BatchExpectPassMatrix> core_mat_ptr;

        // {FRONTLIGHT_intensity, BACKLIGHT_intensity}
        pair<unique_ptr<torch::Tensor>, unique_ptr<torch::Tensor>> _solveMat() const {

            pair<unique_ptr<torch::Tensor>, unique_ptr<torch::Tensor> > FRONTLIGHT_intensity_pair =
                    BatchExpectPassMatrix::ExtractIntensity(this->core_mat_ptr->Solve_FRONTLIGHT());

            pair<unique_ptr<torch::Tensor>, unique_ptr<torch::Tensor> > BACKLIGHT_intensity_pair =
                    BatchExpectPassMatrix::ExtractIntensity(this->core_mat_ptr->Solve_BACKLIGHT());

            return make_pair(std::move(FRONTLIGHT_intensity_pair.second), std::move(BACKLIGHT_intensity_pair.second));
        }
};

// variations

class simpleSimulatedAnnealing : public Optimizer{ // that is freaking dam sit rubbish

    public:
        simpleSimulatedAnnealing(FilaGroup *fila_group, const nlohmann::json &configs,
            const torch::Tensor *target_pic_FRONTLIGHT, const torch::Tensor *target_pic_BACKLIGHT,
            const torch::Tensor *weight_FRONTLIGHT, const torch::Tensor *weight_BACKLIGHT)
            : Optimizer(
                fila_group, configs, target_pic_FRONTLIGHT, target_pic_BACKLIGHT, weight_FRONTLIGHT, weight_BACKLIGHT) {
            _init();
        }

        simpleSimulatedAnnealing(FilaGroup *fila_group, const nlohmann::json &configs,
            const torch::Tensor *target_pic_FRONTLIGHT, const torch::Tensor *target_pic_BACKLIGHT)
            : Optimizer(fila_group, configs, target_pic_FRONTLIGHT, target_pic_BACKLIGHT) {
            _init();
        }

    private:

        void _checkConfig() override;

        void _init() {
            batch_size = target_pic_FRONTLIGHT.numel() / 3; // sizes = {{possibly H, W}, 3}
            layer_size = configs["layer_size"];

            _i_sizes = target_pic_FRONTLIGHT.sizes();
            auto _tmp_sizes = _i_sizes.slice(0, _i_sizes.size() - 1);
            vector<int64_t> _tmp_vec;
            _tmp_vec.insert(_tmp_vec.end(), _tmp_sizes.begin(), _tmp_sizes.end());
            _tmp_vec.push_back(layer_size);
            _o_sizes = _tmp_vec; // should be {{possibly H, W}, L}


            t_f = target_pic_FRONTLIGHT.to(torch::kMPS).flatten(0, 1); // {batch_size, 3}
            t_b = target_pic_BACKLIGHT.to(torch::kMPS).flatten(0, 1);  // {batch_size, 3}
            w_ft = weight_FRONTLIGHT.to(torch::kMPS).flatten(0, 1);    // {batch_size}
            w_bt = weight_BACKLIGHT.to(torch::kMPS).flatten(0, 1);     // {batch_size}


            // init core_mat

            BatchExpectPassMatrix core_mat = BatchExpectPassMatrix(batch_size, layer_size, fila_group);
            this->core_mat_ptr = make_unique<BatchExpectPassMatrix>(core_mat);
        }

        pair<
            pair<unique_ptr<torch::Tensor>, unique_ptr<torch::Tensor>>,
            pair<unique_ptr<torch::Tensor>, unique_ptr<torch::Tensor>> >
        _randDisturb() override;

        // <<FRONTLIGHT_intensity, BACKLIGHT_intensity>, fila_lists.reshape({_o_sizes})>
        pair<pair<unique_ptr<torch::Tensor>, unique_ptr<torch::Tensor>> , unique_ptr<torch::Tensor>> solve() override;

        static torch::Tensor _metropolis_mask(float cur_temperature,
                                              const unique_ptr<torch::Tensor>& pre_loss,
                                              const unique_ptr<torch::Tensor>& cur_loss);

        unique_ptr<torch::Tensor> _loss();

        int batch_size;
        int layer_size;
        at::IntArrayRef _i_sizes;
        at::IntArrayRef _o_sizes;

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