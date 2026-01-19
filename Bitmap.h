//
// Created by xuhang liu on 2026/1/18.
//

#ifndef CFORGE_BITMAP_H
#define CFORGE_BITMAP_H



// maybe we could just define functions and just use torch::Tensor with sizes {H, W, 3(4)}
// (i thought .png could be a useful format to input. 1. lossless 2. extra alpha channel to show the weight of this constraint)

// the final bitmap put should have all the elements in the range of [0, 1]

// the RGB values for the FRONTLIGHT must be directly transformed to [0, 1] by /255
// but the one for the BACKLIGHT should be * by a small number (e.g. 1/1000) as the back-lights always provide high intensity


#endif //CFORGE_BITMAP_H