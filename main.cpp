#include <torch/torch.h>
#include <iostream>
#include <string>

using namespace std;

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

    this->brand = brand;
    this->name = name;
    this->colour = colour;
    this->refractive_index = refractive_index;
    this->extinction_coefficient = extinction_coefficient;
}

torch::Tensor SurfReflct(Filament a, Filament b) {
    return torch::pow((a.refractive_index - b.refractive_index) / (a.refractive_index + b.refractive_index), 2);
}

torch::Tensor LambertEffct(Filament a, float d) {
    return torch::exp( -1 * a.extinction_coefficient * d);
}


static const int MAX_FILA = 105;
static const int MAXBATCH = 10005;
static const int MAXLAYER = 105;
static const int MAXVARIABLES = 215;
static inline const Filament AIR = Filament("Nature", "Air",
    torch::tensor({0, 0, 0}, torch::kUInt8),  // colour
    torch::tensor({1.0, 1.0, 1.0}),           // refractive index
    torch::tensor({0.0, 0.0, 0.0}));          // extinction coefficient

class FilaMatch {

    public:
        int num_fila;
        Filament filaments[MAX_FILA];
        int max_layer;
        float32_t thickness;

    private:
        torch::Tensor P;
        torch::Tensor R;
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
        int num_variables;
        int num_fila;

        torch::Tensor fila_list;
        // a list of fila_index at each layer

        torch::Tensor P;
        torch::Tensor R;
        // 0th is the default material

        BatchExpectPassMatrix(int batch_size, int num_layers, int num_fila, torch::Tensor* P, torch::Tensor* R) {

            this->batch_size = batch_size;
            this->num_layers = num_layers;
            this->num_variables = 2 * (num_layers + 1);
            this->num_fila = num_fila;

            this->P = (*P).clone(); // deep copy
            this->R = (*R).clone();

            this->fila_list = torch::zeros((batch_size, num_layers), torch::kUInt32);

            this->Matrix = torch::zeros((3 * batch_size, num_variables, num_variables), torch::kMPS);
            // 3 for RGB
        }

        torch::Tensor* Solve(torch::Tensor left_input, torch::Tensor right_input);
        void Modify(int batch_index, int layer_index, int target_fila);
        void BatchModify(int* layer_index, int* target_fila);
        // [layer_index] * batch_size, [target_fila] * batch_size
        void Clear();
        void SetMatrix(torch::Tensor* BatchFilaList);
        // a whole integer array of fila_index with shape batch_size * num_layers

    private:
        torch::Tensor Matrix;
        // (3 * batch_size, num_variables, num_variables)
        void _assignElement(int batch_index, char colour, int layer_index_1, char direction_1, int layer_index_2, char direction_2, torch::Tensor modified_value){
            // modified value can be either a single-item tensor or a 3*1 tensor
            // 1*1 for single colour and 3*1 for RGB

            map<char, int> direction_map = {
                {'F', 0},
                {'B', 1}
            };
            map<char, int> colour_map = {
                {'R', 0},
                {'G', 1},
                {'B', 2}
            };
            if (colour != '*') // * for all
                Matrix[batch_index*3 + colour_map[colour]][layer_index_1*2 + direction_map[direction_1]][layer_index_2*2 + direction_map[direction_2]] = modified_value.item();
            else{
                Matrix[batch_index*3 + 0][layer_index_1*2 + direction_map[direction_1]][layer_index_2*2 + direction_map[direction_2]] = modified_value[0];

                Matrix[batch_index*3 + 1][layer_index_1*2 + direction_map[direction_1]][layer_index_2*2 + direction_map[direction_2]] = modified_value[1];

                Matrix[batch_index*3 + 2][layer_index_1*2 + direction_map[direction_1]][layer_index_2*2 + direction_map[direction_2]] = modified_value[2];
            }
        };
};

void BatchExpectPassMatrix::SetMatrix(torch::Tensor* BatchFilaList) {

    this->fila_list = (*BatchFilaList).clone();

    for (int i = 0;i < batch_size;i++)
    for (int j = 0;j < num_layers;j++){

    }

}


int main() {
    torch::Tensor x = torch::rand({2, 3});
    x[0] = torch::tensor({1.0, 2.0, 3.0});
    cout << "Random Tensor:\n" << x << endl;
    return 0;
}