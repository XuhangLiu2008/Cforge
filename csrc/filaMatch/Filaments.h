//
// Created by xuhang liu on 2026/1/15.
//

#ifndef CFORGE_FILAMENTS_H
#define CFORGE_FILAMENTS_H

#include <torch/torch.h>
#include <string>
#include <vector>

using namespace std;



// CLASS: Filament
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

    Filament() {
        brand = "Nature";
        name = "AIR";
        colour = torch::tensor({255, 255, 255}, torch::kUInt8);
        refractive_index = torch::tensor({1.0, 1.0, 1.0});
        extinction_coefficient = torch::tensor({0.0, 0.0, 0.0});
    }

    static torch::Tensor SurfReflct(const Filament &a, const Filament &b);
    static torch::Tensor LambertEffct(const Filament &a, const float d);

    // static const Filament AIR;
};

// CLASS: FilaGroup
class FilaGroup {

    public:
        int num_fila;
        vector<Filament> filaments;
        float32_t thickness;

        torch::Tensor P;
        torch::Tensor R;

        FilaGroup(const int num_fila, const float32_t thickness, const vector<Filament> *filaments) {
            this->num_fila = num_fila;
            this->thickness = thickness; // in millimetre
            this->filaments = *filaments;

            _calculatePnR();
        }

        FilaGroup(const string &filename);

        void save(const string &filename) const;

    private:
        const string FILE_SUFFIX = ".txt";

        void _calculatePnR() {
            P = torch::zeros({num_fila, num_fila, 3});
            R = torch::zeros({num_fila, num_fila, 3});

            for (int i = 0;i < num_fila;i++)
            for (int j = 0;j < num_fila;j++){
                torch::Tensor reflectance = Filament::SurfReflct(filaments[i], filaments[j]);
                P[i][j] = (1 - reflectance) * Filament::LambertEffct(filaments[j], thickness);
                R[i][j] = reflectance * Filament::LambertEffct(filaments[i], thickness);
            }
        };
};


#endif //CFORGE_FILAMENTS_H