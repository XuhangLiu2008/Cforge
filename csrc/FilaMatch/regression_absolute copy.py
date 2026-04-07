# wo ta ma hai mei zuo wan
# bu yao yong


from operator import le

import numpy as np

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import sampling

import pprint

class Filament:

    """

    NOTE: Variables Table
    _________________________________________________________________________________________________
    name | meaning                                                                                  |
    -----+------------------------------------------------------------------------------------------|
    d    | thickness                                                                                |
    K    | absorption coefficient                                                                   |
    S    | scattering coefficient                                                                   |
         |                                                                                          |
    T_KM | transmittance calculated by Kubelka-Munk theory, without considering the base reflection |
    R_KM | reflectance calculated by Kubelka-Munk theory, without considering the base reflection   |
         |                                                                                          |
    T_m  | transmittance after Saunderson correction, which considers the base reflection           |
    R_m  | reflectance after Saunderson correction, which considers the base reflection             |
         |                                                                                          |
    k1   | external reflectance of surface where light enters the filament                          |
    k2   | external reflectance of surface where light leaves the filament                          |
    r1   | internal reflectance of surface where light enters the filament                          |
    r2   | internal reflectance of surface where light leaves the filament                          |
         |                                                                                          |
    k_t  | external reflectance of the top surface                                                  |
    k_b  | external reflectance of the bottom surface                                               |
    r_t  | internal reflectance of the top surface                                                  |
    r_b  | internal reflectance of the bottom surface                                               |
    _________________________________________________________________________________________________

    """

    default_k_t = 0.02
    default_k_b = 0.02
    default_r_t = 0.15
    default_r_b = 0.15

    @staticmethod
    def KMrates(K : np.ndarray, S : np.ndarray, thickness : float) -> tuple[np.ndarray, np.ndarray]:
        # K stands for absorption coefficient, S stands for scattering coefficient
        # T_KM is the penetrate rate, R_KM is the reflectance

        #IMPORTANCT: T_KM and R_KM are the penetrate rate and reflectance that do not consider the base reflection. so it could be used to calculate the circumstance where multiple filaments are stacked

        T_KM = np.zeros(3, dtype=float)
        R_KM = np.zeros(3, dtype=float)

        for i in range(3):
            a = K[i] + S[i]
            sigma = np.sqrt(K[i] * (K[i] + 2 * S[i]))

            # Numerically stable computation
            if sigma < 1e-6:  # sigma ≈ 0
                T_KM[i] = 1.0 / (1.0 + a * thickness)
                R_KM[i] = S[i] * thickness / (1.0 + a * thickness)
            else:
                sigma_x = sigma * thickness
                cosh_val = np.cosh(sigma_x)
                sinh_val = np.sinh(sigma_x)
                D = sigma * cosh_val + a * sinh_val
                T_KM[i] = sigma / D
                R_KM[i] = S[i] * sinh_val / D

        return T_KM, R_KM

    @staticmethod
    def SaundersonCorrection(T_KM : np.ndarray, R_KM : np.ndarray, k1 : float, r1 : float, r2 : float) -> tuple[np.ndarray, np.ndarray]:

        T_m = np.zeros(3, dtype=float)
        R_m = np.zeros(3, dtype=float)

        for i in range(3):
            # Apply Saunderson correction formula
            # T_m = (1 - k1) * (1 - k2) * T_KM / (1 - k2 * R_KM)
            # R_m = k1 + (1 - k1)^2 * (R_KM + k2 * T_KM^2) / (1 - k2 * R_KM)

            denominator = (1.0 - r1 * R_KM[i]) * (1.0 - r2 * R_KM[i]) - r1 * r2 * T_KM[i] ** 2
            T_m[i] = (1.0 - k1) * (1.0 - r2) * T_KM[i] / denominator
            R_m[i] = k1 + (1.0 - k1) * (1.0 - r1) * (R_KM[i] - r2 * (R_KM[i] ** 2 - T_KM[i] ** 2)) / denominator

        return T_m, R_m

    
    @staticmethod
    def RatesInAir(d, K, S, k_t = default_k_t, 
                   k_b = default_k_b, 
                   r_t = default_r_t, 
                   r_b = default_r_b):
        T_KM, R_KM = Filament.KMrates(K, S, d)
        T_m, _ = Filament.SaundersonCorrection(T_KM, R_KM, k_b, r_b, r_t)
        _, R_m = Filament.SaundersonCorrection(T_KM, R_KM, k_t, r_t, r_b)
        return T_m, R_m

    def __init__(self, brand, name,
        absorb_coeff = np.zeros(3, dtype=float),
        scatter_coeff = np.zeros(3, dtype=float),
        icon_colour = None):

        
        self.brand = brand
        self.name = name

        self.icon_colour = icon_colour

        self.absorb_coeff = absorb_coeff
        self.scatter_coeff = scatter_coeff

        self.t_samples = []
        self.r_samples = []

    def sampling(self, T_image_path, R_image_path):
        self.t_samples = sampling.square_sampling(T_image_path)
        self.r_samples = sampling.square_sampling(R_image_path)

    def calculateCoefficients(self, shown = False, color_temp = 4000, gamma = 2.4):
        # samples is a list of [thickness, colour]
        # colour should be np.uint8 array with size 3

        def combinedCoeff(thickness, K_r, S_r, K_g, S_g, K_b, S_b):
            # thickness here should be an array with 6 identical copies
            # to fit all coeffs together
            length = len(thickness) // 6

            K = np.array([K_r, K_g, K_b])
            S = np.array([S_r, S_g, S_b])

            t_res = np.zeros((length, 3))
            r_res = np.zeros((length, 3))

            for i in range(length):
                t_res[i], r_res[i] = Filament.RatesInAir(thickness[i], K, S)

            return list(t_res.T.flatten()) + list(r_res.T.flatten())

        thickness_list = []
        t_rgb_list = [[], [], []]
        r_rgb_list = [[], [], []]

        for i in range(len(self.t_samples)):
            t_sample = self.t_samples[i]
            r_sample = self.r_samples[i]

            if t_sample[0] != r_sample[0]:
                raise ValueError(f"Thickness mismatch between T and R samples at index {i}: {t_sample[0]} vs {r_sample[0]}")
            thickness_list.append(t_sample[0])

            t_rgb_list[0].append(t_sample[1][0] / 255.0)
            t_rgb_list[1].append(t_sample[1][1] / 255.0)
            t_rgb_list[2].append(t_sample[1][2] / 255.0)

            r_rgb_list[0].append(r_sample[1][0] / 255.0)
            r_rgb_list[1].append(r_sample[1][1] / 255.0)
            r_rgb_list[2].append(r_sample[1][2] / 255.0)

        reasonable_guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        bounds = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        # constrain params to physically meaningful ranges

        MAXFEV = int(1e6)

        thickness_arr = np.asarray(thickness_list * 6, dtype=float)
        target_arr = np.asarray(t_rgb_list[0]+t_rgb_list[1]+t_rgb_list[2]+r_rgb_list[0]+r_rgb_list[1]+r_rgb_list[2], dtype=float)

        coefficient, covariance = curve_fit(combinedCoeff, thickness_arr, target_arr, p0=reasonable_guess, bounds=bounds, maxfev=MAXFEV)

        self.absorb_coeff = np.array([coefficient[0], coefficient[2], coefficient[4]])
        self.scatter_coeff = np.array([coefficient[1], coefficient[3], coefficient[5]])

        surface_reflectance = coefficient[6:10]

        print("absorb coeff:", self.absorb_coeff)
        print("scatter coeff:", self.scatter_coeff)
        print("surface reflectance (k_t, k_b, r_t, r_b):", surface_reflectance)

        def display_results_plot():
            
            d_sample = np.asarray(thickness_list, dtype=float)
            t_sample = np.asarray(t_rgb_list, dtype=float)
            r_sample = np.asarray(r_rgb_list, dtype=float)

            d_data = np.linspace(0, np.max(d_sample) * 1.1, 1000)
            t_rgb_data = np.zeros((d_data.shape[0], 3), dtype=float)
            r_rgb_data = np.zeros((d_data.shape[0], 3), dtype=float)

            for i, thickness in enumerate(d_data):
                t_rgb_data[i], r_rgb_data[i] = Filament.RatesInAir(thickness, self.absorb_coeff, self.scatter_coeff, *surface_reflectance)

            t_r_data = t_rgb_data[:, 0]
            t_g_data = t_rgb_data[:, 1]
            t_b_data = t_rgb_data[:, 2]

            r_r_data = r_rgb_data[:, 0]
            r_g_data = r_rgb_data[:, 1]
            r_b_data = r_rgb_data[:, 2]

            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)

            plt.scatter(d_sample, t_sample[0], c = 'r', label='R samples')
            plt.scatter(d_sample, t_sample[1], c = 'g', label='G samples')
            plt.scatter(d_sample, t_sample[2], c = 'b', label='B samples')
            plt.plot(d_data, t_r_data, 'r-', label='R fit')
            plt.plot(d_data, t_g_data, 'g-', label='G fit')
            plt.plot(d_data, t_b_data, 'b-', label='B fit')

            plt.vlines(d_data, [-0.2*np.max(t_r_data)] * 1000, [0] * 1000, np.array(t_rgb_data)/np.max(t_rgb_data))
            plt.text(0.1, -0.1*np.max(t_r_data), f"PENETRATE (*{round(1/np.max(t_rgb_data), 2)})")

            plt.xlabel('Thickness (mm)')
            plt.ylabel('Penetrate Rate')
            plt.xlim(0, np.max(d_sample) * 1.2)
            # plt.legend()
            plt.grid()
            plt.title("Penetrate Rate vs. Thickness")

            plt.subplot(1, 2, 2)

            plt.scatter(d_sample, r_sample[0], c = 'r', label='R samples')
            plt.scatter(d_sample, r_sample[1], c = 'g', label='G samples')
            plt.scatter(d_sample, r_sample[2], c = 'b', label='B samples')
            plt.plot(d_data, r_r_data, 'r-', label='R fit')
            plt.plot(d_data, r_g_data, 'g-', label='G fit')
            plt.plot(d_data, r_b_data, 'b-', label='B fit')

            plt.xlabel('Thickness (mm)')
            plt.ylabel('Reflectance')
            plt.xlim(0, np.max(d_sample) * 1.2)
            # plt.legend()
            plt.grid()
            plt.title("Reflectance vs. Thickness")

            plt.vlines(d_data, [-0.2*np.max(r_r_data)] * 1000, [-0] * 1000, np.array(np.clip(r_rgb_data, 0, 1)))
            plt.text(0.1, -0.1*np.max(r_r_data), "REFLECT")

        def display_prediction(): # 这个还要改，但是不着急
            colour_list = []
            for d in thickness_list:
                colour_list.append(list(Filament.RatesInAir(d, self.absorb_coeff, self.scatter_coeff, 1)[0]))

            max_v = max(max(colour_list))

            for i in range(len(colour_list)):
                colour_list[i] = [thickness_list[i]] + [np.array(colour_list[i]) / max_v * 255]

            # Debug Used Long Outputs
            if DEBUG:
                for i in colour_list:
                    for i_indx in range(len(i[1])):
                        i[1][i_indx] = int((i[1][i_indx]))

                # Brightness
                for i in colour_list:
                    ratio = 0.2
                    i[1][0] += ratio * (255 - i[1][0])
                    i[1][1] += ratio * (255 - i[1][1])
                    i[1][2] += ratio * (255 - i[1][2])
                
                # Green correction
                for i in colour_list:
                    ratio = 0.05
                    i[1][1] += ratio * i[1][1]
                
                # Saturation
                # for i in colour_list:
                #     ratio = 0.9
                #     r, g, b = i[1]

                #     y = 0.2126 * r + 0.7152 * g + 0.0722 * b

                #     r2 = y + ratio * (r - y)
                #     g2 = y + ratio * (g - y)
                #     b2 = y + ratio * (b - y)

                #     i[1][0] = max(0, min(255, round(r2)))
                #     i[1][1] = max(0, min(255, round(g2)))
                #     i[1][2] = max(0, min(255, round(b2)))
                
                red_color_rate = 0.0
                green_color_rate = 0.0
                blue_color_rate = 0.0
                for i in colour_list:
                    red_color_rate += i[1][0] - 0.5 * (i[1][1] + i[1][2])
                    green_color_rate += i[1][1] - 0.5 * (i[1][0] + i[1][2])
                    blue_color_rate += i[1][2] - 0.5 * (i[1][0] + i[1][1])
                red_color_rate = red_color_rate / 16

                print("<<<<<Predicted>>>>>")
                pprint.pprint(colour_list)
                print(red_color_rate, ", ", green_color_rate, ", ", blue_color_rate)

            sampling.display_sample(colour_list)
            plt.title(f"Predicted Colours (*{round(1/max_v, 2)})")

        def display_origin(): # 这个还要改，但是不着急
            sampling.display_square_sample(self.r_samples)
            plt.title("Sampled Colours")

        if shown:
            display_results_plot()
            plt.figure(1)

            # display_prediction()
            # plt.figure(2)

            display_origin()
            plt.figure(2)

            plt.show()

        return 

if __name__ == '__main__':
    DEBUG = True

    image_path = "csrc/FilaMatch/filament02.png"
    test_filament = Filament("test", "test")
    # test_filament.sampling(image_path, image_path)

    t_array = [(0.1, (237.0, 214.0, 116.0)), (0.2, (203.0, 150.0, 23.0)), (0.3, (166.0, 94.0, 0.0)), (0.4, (134.0, 65.0, np.float64(0.0))), (0.5, (109.0, 45.0, np.float64(0.0))), (0.6, (97.0, 32.0, np.float64(0.0))), (0.7, (91.0, 24.0, np.float64(0.0))), (0.8, (86.0, 18.0, np.float64(0.0))), (0.9, (63.0, 12.0, np.float64(0.0))), (1.0, (72.0, 10.0, np.float64(0.0))), (1.1, (67.0, 8.0, np.float64(0.0))), (1.2, (61.0, 6.0, np.float64(0.0))), (1.3, (58.0, 4.0, np.float64(0.0))), (1.4, (46.0, 3.0, np.float64(0.0))), (1.5, (50.0, 2.0, np.float64(0.0))), (1.6, (49.0, 6.0, np.float64(0.00928948029123776)))]
    
    r_array = [(0.1, (12.0, 9.0, 2.0)), (0.2, (22.0, 15.0, 2.0)), (0.3, (35.0, 24.0, 4.0)), (0.4, (35.0, 18.0, 2.0)), (0.5, (38.0, 23.0, 1.0)), (0.6, (40.0, 24.0, 1.0)), (0.7, (42.0, 23.0, 0.0)), (0.8, (42.0, 23.0, 1.0)), (0.9, (42.0, 24.0, 1.0)), (1.0, (45.0, 26.0, 2.0)), (1.1, (44.0, 25.0, 1.0)), (1.2, (44.0, 25.0, 1.0)), (1.3, (44.0, 23.0, 0.0)), (1.4, (45.0, 24.0, 0.0)), (1.5, (46.0, 26.0, 2.0)), (1.6, (44.0, 23.0, 0.0))]

    test_filament.t_samples = t_array
    test_filament.r_samples = r_array

    test_filament.calculateCoefficients(True)

    # Debug Used Long Outputs
    if DEBUG:
        red_color_rate = 0.0
        green_color_rate = 0.0
        blue_color_rate = 0.0
        for i in r_array:
            red_color_rate += i[1][0] - 0.5 * (i[1][1] + i[1][2])
            green_color_rate += i[1][1] - 0.5 * (i[1][0] + i[1][2])
            blue_color_rate += i[1][2] - 0.5 * (i[1][0] + i[1][1])
        red_color_rate = red_color_rate / 16

        print("<<<<<Original>>>>>")
        pprint.pprint(r_array)
        print(red_color_rate, ", ", green_color_rate, ", ", blue_color_rate)

    # r_array = [[0.1, [132, 167, 202]], 
    #            [0.2, [181, 194, 177]], 
    #            [0.3, [194, 208, 195]], 
    #            [0.4, [207, 206, 175]], 
    #            [0.5, [213, 205, 169]], 
    #            [0.6, [217, 205, 163]], 
    #            [0.7, [223, 207, 169]], 
    #            [0.8, [225, 208, 165]], 
    #            [0.9, [224, 203, 161]], 
    #            [1.0, [230, 211, 177]], 
    #            [1.1, [230, 210, 163]], 
    #            [1.2, [231, 207, 160]], 
    #            [1.3, [231, 206, 158]], 
    #            [1.4, [233, 207, 160]], 
    #            [1.5, [238, 217, 190]], 
    #            [1.6, [233, 206, 161]]]