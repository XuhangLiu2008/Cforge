//
// Created by xuhang liu on 2026/1/15.
//

#include "Filaments.h"

#include <torch/torch.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;



// FUNCITONS: Filament
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

torch::Tensor Filament::SurfReflct(const Filament &a, const Filament &b) {
    return torch::pow((a.refractive_index - b.refractive_index) / (a.refractive_index + b.refractive_index), 2);
}

torch::Tensor Filament::LambertEffct(const Filament &a, const float d) {
    auto res = torch::exp( -1 * a.extinction_coefficient * d);
    return res;
}

// const Filament AIR = Filament("Nature", "AIR",
//         torch::tensor({255, 255, 255}, torch::kUInt8),
//         torch::tensor({1.0, 1.0, 1.0}),
//         torch::tensor({0.0, 0.0, 0.0}));


// FUNCTIONS: FilaGroup
FilaGroup::FilaGroup(const string &filename) {
    if (filename.substr(filename.length()-FILE_SUFFIX.length(), FILE_SUFFIX.length()) != FILE_SUFFIX) {
        throw format_error("The suffix of the file should be "+FILE_SUFFIX);
    }

    ifstream inFile;
    inFile.open(filename);

    if (!inFile.is_open()) {
        throw runtime_error("Could not open file "+filename);
    }

    inFile >> num_fila;
    inFile >> thickness;

    for (int i = 0;i < num_fila;i++) {

        string brand, name;
        inFile >> brand >> name;

        int colour[3];
        inFile >> colour[0] >> colour[1] >> colour[2];

        float32_t refractive_index[3];
        inFile >> refractive_index[0] >> refractive_index[1] >> refractive_index[2];

        float32_t extinction_coefficient[3];
        inFile >> extinction_coefficient[0] >> extinction_coefficient[1] >> extinction_coefficient[2];

        this->filaments.push_back(Filament(brand, name,
            torch::tensor({colour[0], colour[1], colour[2]}, torch::kUInt8),
            torch::tensor({refractive_index[0], refractive_index[1], refractive_index[2]}),
            torch::tensor({extinction_coefficient[0], extinction_coefficient[1], extinction_coefficient[2]})));

    }

    this->P = torch::zeros({num_fila, num_fila, 3});
    this->R = torch::zeros({num_fila, num_fila, 3});

    for (int i = 0;i < num_fila;i++) {
        for (int j = 0;j < num_fila;j++) {
            float32_t num[3];
            inFile >> num[0] >> num[1] >> num[2];
            P[i][j] = torch::tensor({num[0], num[1], num[2]}, torch::kFloat32);
        }
    }

    for (int i = 0;i < num_fila;i++) {
        for (int j = 0;j < num_fila;j++) {
            float32_t num[3];
            inFile >> num[0] >> num[1] >> num[2];
            R[i][j] = torch::tensor({num[0], num[1], num[2]}, torch::kFloat32);
        }
    }
};

void FilaGroup::save(const string &filename) const {
    if (filename.substr(filename.length()-FILE_SUFFIX.length(), FILE_SUFFIX.length()) != FILE_SUFFIX) {
        throw format_error("The suffix of the file should be "+FILE_SUFFIX);
    }

    ofstream outFile;
    outFile.open(filename);

    if (!outFile.is_open()) {
        throw runtime_error("Could not open file "+filename);
    }

    outFile << num_fila << endl;
    outFile << thickness << endl;

    for (int i = 0;i < num_fila;i++) {
        outFile << filaments[i].brand << ' ' << filaments[i].name << ' ';
        outFile << filaments[i].colour[0].item() << ' ' << filaments[i].colour[1].item() << ' ' << filaments[i].colour[2].item() << ' ';
        outFile << filaments[i].refractive_index[0].item() << ' ' << filaments[i].refractive_index[1].item() << ' ' << filaments[i].refractive_index[2].item() << ' ';
        outFile << filaments[i].extinction_coefficient[0].item() << ' ' << filaments[i].extinction_coefficient[1].item() << ' ' << filaments[i].extinction_coefficient[2].item() << ' ';
        outFile << endl;
    }

    for (int i = 0;i < num_fila;i++) {
        for (int j = 0;j < num_fila;j++) {
            outFile << P[i][j][0].item() << ' ' << P[i][j][1].item() << ' ' << P[i][j][2].item() << ' ';
        }
        outFile << endl;
    }

    for (int i = 0;i < num_fila;i++) {
        for (int j = 0;j < num_fila;j++) {
            outFile << R[i][j][0].item() << ' ' << R[i][j][1].item() << ' ' << R[i][j][2].item() << ' ';
        }
        outFile << endl;
    }

    outFile.close();
};