#include <torch/torch.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

/*
MARK: KEY PRINCIPLE

----AIR--->] [<--Fila1-->] [<--Fila2-->] [<--........-->] [<Fila(n-1)>] [<---AIR----
            |             |             |                |             |
       E_f0 |        E_f1 |        E_f2 |                |    E_f(n-1) |        E_fn
----------> | ----------> | ----------> | ----......---> | ----------> | ---------->
            |             |             |                |             |
<---------- | <---------- | <---------- | <---......---- | <---------- | <----------
E_b0        | E_b1        | E_b2        |                | E_b(n-1)    | E_bn
            |             |             |                |             |


E : the expectation of the number of passing for one photon

E_f0, E_bn given
E_b0, E_fn needed (exactly the same as the ratio of the intensity)

modeled by Markov chain, we get:

E_fi = P[i-1][i] * E_f(i-1) + R[i][i-1] * E_bi
E_bi = P[i+1][i] * E_b(i+1) + R[i][i+1] * E_fi

for each i


where:

r[i][j] = ( (n_i - n_j) / (n_i + n_j) ) ** 2  # reflected ratio at the surface
P[i][j] = (1 - r[i][j]) * exp(-K[j] * d)
R[i][j] = r[i][j] * exp(-K[i] * d)


get E_b0 and E_fn by solving the simultaneous equations

*/

struct Filament {
    string brand;
    string name;
    torch::Tensor colour; // set by user, does not impact calculation
    torch::Tensor refractive_index;
    torch::Tensor extinction_coefficient;
    Filament(string brand,
    string name,
    torch::Tensor colour,
    torch::Tensor refractive_index,
    torch::Tensor extinction_coefficient);
};

Filament::Filament(string brand, string name,
    torch::Tensor colour,
    torch::Tensor refractive_index,
    torch::Tensor extinction_coefficient) {

    this->brand = std::move(brand);
    this->name = std::move(name);
    this->colour = std::move(colour);
    this->refractive_index = std::move(refractive_index);
    this->extinction_coefficient = std::move(extinction_coefficient);
}

torch::Tensor SurfReflct(const Filament &a, const Filament &b) {
    return torch::pow((a.refractive_index - b.refractive_index) / (a.refractive_index + b.refractive_index), 2);
}

torch::Tensor LambertEffct(const Filament &a, const float d) {
    return torch::exp( -1 * a.extinction_coefficient * d);
}


static inline const Filament AIR = Filament("Nature", "Air",
    torch::tensor({0, 0, 0}, torch::kUInt8),  // colour
    torch::tensor({1.0, 1.0, 1.0}),           // refractive index
    torch::tensor({0.0, 0.0, 0.0}));          // extinction coefficient



class FilaGroup {

    public:
        int num_fila;
        vector<Filament> filaments;
        float32_t thickness;

        torch::Tensor P;
        torch::Tensor R;

        FilaGroup(const int num_fila, const float32_t thickness, const vector<Filament> *filaments) {
            this->num_fila = num_fila;
            this->thickness = thickness;
            this->filaments = *filaments;
        }

    private:

        void _calculatePnR(){
            P = torch::zeros((num_fila, num_fila, 3));
            R = torch::zeros((num_fila, num_fila, 3));

            for (int i = 0;i < num_fila;i++)
            for (int j = 0;j < num_fila;j++){
                torch::Tensor reflectance = SurfReflct(filaments[i], filaments[j]);
                P[i][j] = (1 - reflectance) * LambertEffct(filaments[j], thickness);
                R[i][j] = reflectance * LambertEffct(filaments[i], thickness);
            }
        };
};



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
                              FilaGroup* filaments) {

            this->batch_size = batch_size;
            this->num_layers = num_layers;
            this->num_variables = 2 * num_layers;

            this->filaments = filaments;

            this->P = &(this->filaments->P);
            this->R = &(this->filaments->R);

            this->fila_list = torch::zeros((batch_size, num_layers), torch::kUInt32);

            this->Matrix = torch::zeros((batch_size * 3, num_variables, num_variables), torch::kMPS);
            // 3 for RGB

            for (int batch_index = 0;batch_index < batch_size;batch_index++)
                for (int layer_index = 0;layer_index < num_layers;layer_index++) {

                    _assignElement(batch_index, '*', layer_index, 'f', layer_index, 'f',
                        torch::tensor({-1.0, -1.0, -1.0}));
                    _assignElement(batch_index, '*', layer_index, 'b', layer_index, 'b',
                        torch::tensor({-1.0, -1.0, -1.0}));
                }

            this->BACKLIGHT_CONST = torch::zeros((batch_size * 3, num_variables), torch::kFloat32).to(torch::kMPS);
            this->BACKLIGHT_CONST.index_put_({torch::indexing::Slice(), 0}, -1.0); // E_f0

            this->FRONTLIGHT_CONST = torch::zeros((batch_size * 3, num_variables), torch::kFloat32).to(torch::kMPS);
            this->FRONTLIGHT_CONST.index_put_({torch::indexing::Slice(), num_variables-1}, -1.0); // E_bn
        }

        torch::Tensor* Solve(const torch::Tensor* left_input, const torch::Tensor* right_input);
        torch::Tensor* Solve_BACKLIGHT() {
            torch::Tensor res = torch::linalg_solve(Matrix, BACKLIGHT_CONST);
            return &res;
        };
        torch::Tensor* Solve_FRONTLIGHT() {
            torch::Tensor res = torch::linalg_solve(Matrix, FRONTLIGHT_CONST);
            return &res;
        }

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

        // _updateLine function should be called after the fila_list is modified
        void _updateLine(const int batch_index, const int layer_index, const char direction) {
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

        torch::Tensor BACKLIGHT_CONST;
        torch::Tensor FRONTLIGHT_CONST;
};

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
}

void BatchExpectPassMatrix::Clear() {
    // set all fila to default (0)
    fila_list = torch::zeros_like(fila_list);
    SetMatrix(&fila_list);
}

torch::Tensor* BatchExpectPassMatrix::Solve(const torch::Tensor* left_input, const torch::Tensor* right_input) {
    // the shape of the left/right_input should be (batch_size*3)
    torch::Tensor Constants = torch::zeros((batch_size * 3, num_variables));

    for (int i = 0;i < (batch_size*3);i++) {

        // E_f0
        Constants[i][0] = -(*left_input)[i];
        // negative as the coefficients on the main diagonal are all -1 instead of 1

        // E_bn
        Constants[i][num_variables-1] = -(*right_input)[num_variables-1];
    }

    torch::Tensor res = torch::linalg_solve(Matrix, Constants.to(torch::kMPS));
    return &res;

}

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

void BatchExpectPassMatrix::BatchModify(const int *layer_index, const int *target_fila) {
    // layer_index : integer 1d array with length batch_size
    // target_fila : integer 1d array with length batch_size

    for (int batch_index = 0;batch_index < batch_size;batch_index++) {
        if (layer_index[batch_index] == -1) continue;
        Modify(batch_index, layer_index[batch_index], target_fila[batch_index]);
    }
}



class Optimizer {
    public:

};



int main() {
    torch::Tensor x = torch::rand({2, 3});
    x[0] = torch::tensor({1.0, 2.0, 3.0});
    cout << "Random Tensor:\n" << x << endl;
    return 0;
}