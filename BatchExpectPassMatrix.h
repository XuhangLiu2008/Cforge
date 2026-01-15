//
// Created by xuhang liu on 2026/1/15.
//

#ifndef CFORGE_BATCHEXPECTPASSMATRIX_H
#define CFORGE_BATCHEXPECTPASSMATRIX_H

#include <torch/torch.h>
#include "Filaments.h"
#include <memory>

using namespace std;



class BatchExpectPassMatrix {

    public:
        int batch_size;
        int num_layers;
        // the filaments at the first/last layer should be air
        int num_variables;

        torch::Tensor fila_list;
        // a list of fila_index at each layer with shape (batch_size, num_layers)

        FilaGroup* filaments;

        // init
        BatchExpectPassMatrix(const int batch_size,
                              const int num_layers,
                              FilaGroup* filaments);

        unique_ptr<torch::Tensor> Solve(const torch::Tensor* left_input, const torch::Tensor* right_input);
        unique_ptr<torch::Tensor> Solve_BACKLIGHT();
        unique_ptr<torch::Tensor> Solve_FRONTLIGHT();

        void Modify(int batch_index, int layer_index, int target_fila);
        void BatchModify(const int* layer_index, const int* target_fila);
        // [layer_index] * batch_size, [target_fila] * batch_size
        void Clear();
        void SetMatrix(const torch::Tensor* BatchFilaList);
        // a whole integer array of fila_index with shape batch_size * num_layers
        static pair<torch::Tensor*, torch::Tensor*> ExtractIntensity(const torch::Tensor* BatchExpectPass) {
            // BatchExpectPass: shape(batch_size * 3, num_variables)
            torch::Tensor left_output = BatchExpectPass->index({torch::indexing::Slice(), 1}); // E_b0
            torch::Tensor right_output = BatchExpectPass->index({torch::indexing::Slice(), BatchExpectPass->size(1)-1}); // E_fn
            return make_pair(&left_output, &right_output);
        };

    private:

        torch::Tensor* P;
        torch::Tensor* R;
        // 0th is the default material
        // most times should be air

        torch::Tensor Matrix;
        // (3 * batch_size, num_variables, num_variables)

        void _assignElement(const int batch_index,
                            const char colour,
                            const int layer_index_1,
                            const char direction_1,
                            const int layer_index_2,
                            const char direction_2,
                            const torch::Tensor &modified_value);

        // _updateLine function should be called after the fila_list is modified
        void _updateLine(const int batch_index, const int layer_index, const char direction);

        torch::Tensor BACKLIGHT_CONST;
        torch::Tensor FRONTLIGHT_CONST;
};



#endif //CFORGE_BATCHEXPECTPASSMATRIX_H