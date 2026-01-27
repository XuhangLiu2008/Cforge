//
// Created by xuhang liu on 2026/1/15.
//

#include "BatchExpectPassMatrix.h"
#include <memory>
#include <torch/torch.h>



void BatchExpectPassMatrix::_assignElement(const int batch_index,
                    const char colour,
                    const int layer_index_1,
                    const char direction_1,
                    const int layer_index_2,
                    const char direction_2,
                    const torch::Tensor &modified_value) {

    // modified value can be either a single-item tensor or a 3*1 tensor
    // 1*1 for single colour and 3*1 for RGB

    map<char, int> direction_map = {
        {'f', 0},
        {'b', 1}
    };
    map<char, int> colour_map = {
        {'r', 0},
        {'g', 1},
        {'b', 2}
    };
    if (colour != '*') // * for all
        Matrix[batch_index*3 + colour_map[colour]][layer_index_1*2 + direction_map[direction_1]][layer_index_2*2 + direction_map[direction_2]] = modified_value.item();
    else{
        Matrix[batch_index*3 + 0][layer_index_1*2 + direction_map[direction_1]][layer_index_2*2 + direction_map[direction_2]] = modified_value[0];

        Matrix[batch_index*3 + 1][layer_index_1*2 + direction_map[direction_1]][layer_index_2*2 + direction_map[direction_2]] = modified_value[1];

        Matrix[batch_index*3 + 2][layer_index_1*2 + direction_map[direction_1]][layer_index_2*2 + direction_map[direction_2]] = modified_value[2];
    }
};

void BatchExpectPassMatrix::_updateLine(const int batch_index, const int layer_index, const char direction) {
    if (direction == 'f') {

        if (layer_index == 0) return;

        // E_fi = P[i-1][i] * E_f(i-1) + R[i][i-1] * E_bi
        _assignElement(batch_index, '*', layer_index, 'f', layer_index-1, 'f',
            (*P)[fila_list[batch_index][layer_index-1]][fila_list[batch_index][layer_index]]);
        _assignElement(batch_index, '*', layer_index, 'f', layer_index, 'b',
            (*R)[fila_list[batch_index][layer_index]][fila_list[batch_index][layer_index-1]]);
    }
    else if (direction == 'b') {

        if (layer_index == num_layers-1) return;

        // E_bi = P[i+1][i] * E_b(i+1) + R[i][i+1] * E_fi
        _assignElement(batch_index, '*', layer_index, 'b', layer_index+1, 'b',
            (*P)[fila_list[batch_index][layer_index+1]][fila_list[batch_index][layer_index]]);
        _assignElement(batch_index, '*', layer_index, 'b', layer_index, 'f',
            (*R)[fila_list[batch_index][layer_index]][fila_list[batch_index][layer_index+1]]);
    }
    else throw std::invalid_argument("Invalid direction");
}



// init
BatchExpectPassMatrix::BatchExpectPassMatrix(const int batch_size,
                      const int num_layers,
                      FilaGroup* filaments) {

    this->batch_size = batch_size;
    this->num_layers = num_layers;
    this->num_variables = 2 * num_layers;

    this->filaments = filaments;

    this->P = &(this->filaments->P);
    this->R = &(this->filaments->R);

    this->fila_list = torch::zeros({batch_size, num_layers}, torch::kUInt32);

    this->Matrix = torch::zeros({batch_size * 3, num_variables, num_variables}, torch::kMPS);
    // 3 for RGB

    for (int batch_index = 0;batch_index < batch_size;batch_index++)
        for (int layer_index = 0;layer_index < num_layers;layer_index++) {

            _assignElement(batch_index, '*', layer_index, 'f', layer_index, 'f',
                torch::tensor({-1.0, -1.0, -1.0}));
            _assignElement(batch_index, '*', layer_index, 'b', layer_index, 'b',
                torch::tensor({-1.0, -1.0, -1.0}));
        }

    this->BACKLIGHT_CONST = torch::zeros({batch_size * 3, num_variables}, torch::kFloat32).to(torch::kMPS);
    this->BACKLIGHT_CONST.index_put_({torch::indexing::Slice(), 0}, -1.0); // E_f0

    this->FRONTLIGHT_CONST = torch::zeros({batch_size * 3, num_variables}, torch::kFloat32).to(torch::kMPS);
    this->FRONTLIGHT_CONST.index_put_({torch::indexing::Slice(), num_variables-1}, -1.0); // E_bn
}

void BatchExpectPassMatrix::SetMatrix(const torch::Tensor* BatchFilaList) {
    this->fila_list = BatchFilaList->clone();

    for (int batch_index = 0;batch_index < batch_size;batch_index++)
        for (int layer_index = 1;layer_index < num_layers;layer_index++) {

            _updateLine(batch_index, layer_index, 'f');
        }

    for (int batch_index = 0;batch_index < batch_size;batch_index++)
        for (int layer_index = 0;layer_index < (num_layers-1);layer_index++) {

            _updateLine(batch_index, layer_index, 'b');
        }

    // for (int batch_index = 0;batch_index < batch_size*3;batch_index++) {
    //     cout << "batch_index = " << batch_index << endl;
    //     torch::Tensor slic = Matrix[batch_index].cpu();
    //     for (int i = 0;i < num_variables;i++) {
    //         for (int j = 0;j < num_variables;j++) {
    //             cout << slic[i][j].item() << ' ';
    //         }
    //         cout << endl;
    //     }
    //     cout << endl;
    // }

    return ;
}

void BatchExpectPassMatrix::Clear() {
    // set all fila to default (0)
    fila_list = torch::zeros_like(fila_list);
    SetMatrix(&fila_list);
}

unique_ptr<torch::Tensor> BatchExpectPassMatrix::Solve(const torch::Tensor* left_input, const torch::Tensor* right_input) {
    // the shape of the left/right_input should be (batch_size*3)
    torch::Tensor Constants = torch::zeros({batch_size * 3, num_variables});

    for (int i = 0;i < (batch_size*3);i++) {

        // E_f0
        Constants[i][0] = -(*left_input)[i];
        // negative as the coefficients on the main diagonal are all -1 instead of 1

        // E_bn
        Constants[i][num_variables-1] = -(*right_input)[num_variables-1];
    }

    unique_ptr<torch::Tensor> res = make_unique<torch::Tensor>(torch::linalg_solve(Matrix, Constants.to(torch::kMPS)));
    return res;

}

unique_ptr<torch::Tensor> BatchExpectPassMatrix::Solve_BACKLIGHT() {
    unique_ptr<torch::Tensor> res = make_unique<torch::Tensor>(torch::linalg_solve(Matrix, BACKLIGHT_CONST));
    return res;
};

unique_ptr<torch::Tensor> BatchExpectPassMatrix::Solve_FRONTLIGHT() {
    unique_ptr<torch::Tensor> res = make_unique<torch::Tensor>(torch::linalg_solve(Matrix, FRONTLIGHT_CONST));
    return res;
};

void BatchExpectPassMatrix::Modify(const int batch_index, const int layer_index, const int target_fila) {

    if (layer_index == 0 || layer_index == num_layers - 1) {
        throw runtime_error("The modification of the first/last layer is not expected.");
    }

    fila_list[batch_index][layer_index] = target_fila;

    // E_fi = P[i-1][i] * E_f(i-1) + R[i][i-1] * E_bi
    // E_bi = P[i+1][i] * E_b(i+1) + R[i][i+1] * E_fi
    // for each i

    // all the equations that P[i][] / P[][i] / R[i][] / R[][i] involves in (needed to update):
    do {
    // E_b(i-1) = P[i][i-1] * E_bi + R[i-1][i] * E_f(i-1)
    _updateLine(batch_index, layer_index-1, 'f');

    // E_fi = P[i-1][i] * E_f(i-1) + R[i][i-1] * E_bi
    _updateLine(batch_index, layer_index, 'f');
    // E_bi = P[i+1][i] * E_b(i+1) + R[i][i+1] * E_fi
    _updateLine(batch_index, layer_index, 'b');

    // E_f(i+1) = P[i][i+1] * E_fi + R[i+1][i] * E_b(i+1)
    _updateLine(batch_index, layer_index+1, 'f');
    } while (false);

}

void BatchExpectPassMatrix::BatchModify(const torch::Tensor *layer_index, const torch::Tensor *target_fila) {
    // layer_index : integer 1d array with length batch_size
    // target_fila : integer 1d array with length batch_size

    for (int batch_index = 0;batch_index < batch_size;batch_index++) {
        if ((*layer_index)[batch_index].item<int>() == -1) continue;
        Modify(batch_index, (*layer_index)[batch_index].item<int>(),
            (*target_fila)[batch_index].item<int>());
    }
}

void BatchExpectPassMatrix::BatchModify(const unique_ptr<torch::Tensor> layer_index, const unique_ptr<torch::Tensor> target_fila) {
    // layer_index : integer 1d array with length batch_size
    // target_fila : integer 1d array with length batch_size

    for (int batch_index = 0;batch_index < batch_size;batch_index++) {
        if ((*layer_index)[batch_index].item<int>() == -1) continue;
        Modify(batch_index, (*layer_index)[batch_index].item<int>(), (*target_fila)[batch_index].item<int>());
    }
}