from operator import le

import numpy as np

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import sampling

import pprint


class Filament:

    refra_index = 1.65
    k1 = 0.11
    k2 = 0.65

    @staticmethod
    def KMrates(K : np.ndarray, S : np.ndarray, thickness : float) -> tuple[np.ndarray, np.ndarray]:
        # K stands for absorption coefficient, S stands for scattering coefficient
        # T_KM is the penetrate rate, R_KM is the reflectance

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
    def SaundersonCorrection(T_KM : np.ndarray, R_KM : np.ndarray, k1 : float, k2 : float) -> tuple[np.ndarray, np.ndarray]:
        # k1 is the reflectance of the surface, k2 is the reflectance of the inner boundary

        T_m = np.zeros(3, dtype=float)
        R_m = np.zeros(3, dtype=float)

        for i in range(3):
            # Apply Saunderson correction formula
            # T_m = (1 - k1) * (1 - k2) * T_KM / (1 - k2 * R_KM)
            # R_m = k1 + (1 - k1)^2 * (R_KM + k2 * T_KM^2) / (1 - k2 * R_KM)

            denominator = 1.0 - k2 * R_KM[i]
            T_m[i] = (1.0 - k1) * (1.0 - k2) * T_KM[i] / denominator
            R_m[i] = k1 + (1.0 - k1) ** 2 * (R_KM[i] + k2 * T_KM[i] ** 2) / denominator

        return T_m, R_m

    @staticmethod
    def RatesInAir(thickness, absorb_coeff, scatter_coeff, enlarge_factor):
        T_KM, R_KM = Filament.KMrates(absorb_coeff, scatter_coeff, thickness)
        T_m, R_m = Filament.SaundersonCorrection(T_KM, R_KM, Filament.k1, Filament.k2)
        return enlarge_factor * T_m, enlarge_factor * R_m

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
        self.t_samples = sampling.sampling(T_image_path)
        self.r_samples = sampling.sampling(R_image_path)
    
    @staticmethod
    def inverseGamma(x):
        x = x / 255.0
        if x <= 0.04045:
            return x / 12.92
        else:
            return ((x + 0.055) / 1.055) ** 2.4

    # R_temp2coff = {4000 : 1.8}
    # G_temp2coff = {4000 : 1.0}
    # B_temp2coff = {4000 : 1.4}

    R_temp2coff = {4000 : 1.0}
    G_temp2coff = {4000 : 1.0}
    B_temp2coff = {4000 : 1.0}

    @staticmethod
    def RGB2RelativeIntensity(color, gamma = 2.4, color_temp = 4000):
        return np.array([Filament.inverseGamma(color[0]) / Filament.R_temp2coff[color_temp],
                         Filament.inverseGamma(color[1]) / Filament.G_temp2coff[color_temp],
                         Filament.inverseGamma(color[2]) / Filament.B_temp2coff[color_temp]])

    def calculateCoefficients(self, shown = False, color_temp = 4000, gamma = 2.4):
        # samples is a list of [thickness, colour]
        # colour should be np.uint8 array with size 3

        def combinedCoeff(thickness, K_r, S_r, K_g, S_g, K_b, S_b, t_enlarge_factor_r, t_enlarge_factor_g, t_enlarge_factor_b, r_enlarge_factor_r, r_enlarge_factor_g, r_enlarge_factor_b):
            # thickness here should be an array with 6 identical copies
            # to fit all coeffs together
            length = len(thickness) // 6

            K = np.array([K_r, K_g, K_b])
            S = np.array([S_r, S_g, S_b])

            t_res = np.zeros((length, 3))
            r_res = np.zeros((length, 3))

            t_enlarge_factor = np.array([t_enlarge_factor_r, t_enlarge_factor_g, t_enlarge_factor_b], dtype=float)
            r_enlarge_factor = np.array([r_enlarge_factor_r, r_enlarge_factor_g, r_enlarge_factor_b], dtype=float)

            for i in range(length):
                t_res[i], _ = Filament.RatesInAir(thickness[i], K, S, t_enlarge_factor)
                _, r_res[i] = Filament.RatesInAir(thickness[i], K, S, r_enlarge_factor)

            return list(t_res.T.flatten()) + list(r_res.T.flatten())

        thickness_list = []

        t_r_list = []
        t_g_list = []
        t_b_list = []

        for sample in self.t_samples:
            thickness_list.append(sample[0])
            intensity = Filament.RGB2RelativeIntensity(sample[1], gamma=gamma, color_temp=color_temp)
            t_r_list.append(intensity[0])
            t_g_list.append(intensity[1])
            t_b_list.append(intensity[2])

        r_r_list = []
        r_g_list = []
        r_b_list = []

        for sample in self.r_samples:

            intensity = Filament.RGB2RelativeIntensity(sample[1], gamma=gamma, color_temp=color_temp)
            r_r_list.append(intensity[0])
            r_g_list.append(intensity[1])
            r_b_list.append(intensity[2])

        reasonable_guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
        bounds = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [10, 10, 10, 10, 10, 10, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        # constrain params to physically meaningful ranges

        MAXFEV = int(1e6)

        thickness_arr = np.asarray(thickness_list * 6, dtype=float)
        target_arr = np.asarray((t_r_list + t_g_list + t_b_list) + 
                                (r_r_list + r_g_list + r_b_list), dtype=float)

        coefficient, covariance = curve_fit(combinedCoeff, thickness_arr, target_arr, p0=reasonable_guess, bounds=bounds, maxfev=MAXFEV)

        r_coefficient = np.array([coefficient[0], coefficient[1], coefficient[6], coefficient[9]])
        g_coefficient = np.array([coefficient[2], coefficient[3], coefficient[7], coefficient[10]])
        b_coefficient = np.array([coefficient[4], coefficient[5], coefficient[8], coefficient[11]])

        self.absorb_coeff = np.array([r_coefficient[0], g_coefficient[0], b_coefficient[0]])
        self.scatter_coeff = np.array([r_coefficient[1], g_coefficient[1], b_coefficient[1]])
        t_enlarge_factor = np.array([r_coefficient[2], g_coefficient[2], b_coefficient[2]])
        r_enlarge_factor = np.array([r_coefficient[3], g_coefficient[3], b_coefficient[3]])

        if self.icon_colour is None:
            tmp = (Filament.RatesInAir(100, self.absorb_coeff, self.scatter_coeff, 1))[1]
            self.icon_colour = np.uint8(tmp * 255)

        print("R coef:", r_coefficient)
        print("G coef:", g_coefficient)
        print("B coef:", b_coefficient)
        print("absorb coeff:", self.absorb_coeff)
        print("scatter coeff:", self.scatter_coeff)
        print("icon colour:", self.icon_colour)

        print("t enlarge factors:", t_enlarge_factor)
        print("r enlarge factors:", r_enlarge_factor)

        def display_results_plot():
            
            d_sample = np.asarray(thickness_list, dtype=float)

            t_r_sample = np.asarray(t_r_list, dtype=float) / t_enlarge_factor[0]
            t_g_sample = np.asarray(t_g_list, dtype=float) / t_enlarge_factor[1]
            t_b_sample = np.asarray(t_b_list, dtype=float) / t_enlarge_factor[2]

            r_r_sample = np.asarray(r_r_list, dtype=float) / r_enlarge_factor[0]
            r_g_sample = np.asarray(r_g_list, dtype=float) / r_enlarge_factor[1]
            r_b_sample = np.asarray(r_b_list, dtype=float) / r_enlarge_factor[2]

            d_data = np.linspace(0, np.max(d_sample) * 1.1, 1000)

            t_rgb_data = np.zeros((d_data.shape[0], 3), dtype=float)
            r_rgb_data = np.zeros((d_data.shape[0], 3), dtype=float)

            for i, thickness in enumerate(d_data):
                t_rgb_data[i], r_rgb_data[i] = Filament.RatesInAir(thickness, self.absorb_coeff, self.scatter_coeff, 1)

            t_r_data = t_rgb_data[:, 0]
            t_g_data = t_rgb_data[:, 1]
            t_b_data = t_rgb_data[:, 2]

            r_r_data = r_rgb_data[:, 0]
            r_g_data = r_rgb_data[:, 1]
            r_b_data = r_rgb_data[:, 2]

            plt.scatter(d_sample, t_r_sample, c = 'r', label='R samples')
            plt.scatter(d_sample, t_g_sample, c = 'g', label='G samples')
            plt.scatter(d_sample, t_b_sample, c = 'b', label='B samples')
            plt.plot(d_data, t_r_data, 'r-', label='R fit')
            plt.plot(d_data, t_g_data, 'g-', label='G fit')
            plt.plot(d_data, t_b_data, 'b-', label='B fit')

            plt.scatter(d_sample, r_r_sample, c = 'r', label='R samples')
            plt.scatter(d_sample, r_g_sample, c = 'g', label='G samples')
            plt.scatter(d_sample, r_b_sample, c = 'b', label='B samples')
            plt.plot(d_data, r_r_data, 'r-', label='R fit')
            plt.plot(d_data, r_g_data, 'g-', label='G fit')
            plt.plot(d_data, r_b_data, 'b-', label='B fit')

            plt.xlabel('Thickness (mm)')
            plt.ylabel('Penetrate Rate / Reflectance')
            plt.ylim(-0.2, 0.5)
            plt.xlim(0, np.max(d_sample) * 1.2)
            plt.legend()

            x_sp = np.linspace(0, np.max(d_sample) * 1.2, 1000)
            t = []
            r = []
            
            for d in x_sp:
                nt, nr = Filament.RatesInAir(d, self.absorb_coeff, self.scatter_coeff, 1)
                t.append([min(1, nt[0]), min(1, nt[1]), min(1, nt[2])])
                r.append([min(1, nr[0]), min(1, nr[1]), min(1, nr[2])])

            plt.vlines(x_sp, [-0.1] * 1000, [0] * 1000, np.array(t)/max(max(t)))
            plt.vlines(x_sp, [-0.2] * 1000, [-0.1] * 1000, np.array(r))

            plt.text(0.1, -0.06, f"PENETRATE (*{round(1/max(max(t)), 2)})")
            plt.text(0.1, -0.16, "REFLECT")

            plt.grid()

            plt.title("Penetrate Rate / Reflectance vs. Thickness")

        def display_prediction(): # 这个还要改，但是不着急
            colour_list = []
            for d in thickness_list:
                colour_list.append(list(Filament.RatesInAir(d, self.absorb_coeff, self.scatter_coeff, 1)[0]))
            
            print(t_enlarge_factor)
            print(r_enlarge_factor)

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

        def display_origin():
            sampling.display_sample(self.samples)
            plt.title("Sampled Colours")

        if shown:
            display_results_plot()
            plt.figure(1)

            display_prediction()
            plt.figure(2)

            display_origin()
            plt.figure(3)

            plt.show()

        return 

if __name__ == '__main__':
    DEBUG = True

    image_path = "csrc/FilaMatch/filament02.png"
    test_filament = Filament("test", "test")
    test_filament.sampling(image_path)

    array = [[0.1, [228, 215, 207]], [0.2, [231, 205, 156]], [0.3, [216, 177, 102]], [0.4, [219, 165, 79]], [0.5, [214, 145, 68]], [0.6, [201, 129, 53]], [0.7, [199, 120, 51]], [0.8, [203, 112, 49]], [0.9, [198, 103, 45]], [1.0, [192, 94, 42]], [1.1, [187, 87, 38]], [1.2, [182, 79, 37]], [1.3, [181, 75, 36]], [1.4, [176, 69, 35]], [1.5, [172, 66, 33]], [1.6, [166, 58, 29]]]
    print(array == test_filament.samples)

    test_filament.calculateCoefficients(True)

    # Debug Used Long Outputs
    if DEBUG:
        red_color_rate = 0.0
        green_color_rate = 0.0
        blue_color_rate = 0.0
        for i in array:
            red_color_rate += i[1][0] - 0.5 * (i[1][1] + i[1][2])
            green_color_rate += i[1][1] - 0.5 * (i[1][0] + i[1][2])
            blue_color_rate += i[1][2] - 0.5 * (i[1][0] + i[1][1])
        red_color_rate = red_color_rate / 16

        print("<<<<<Original>>>>>")
        pprint.pprint(array)
        print(red_color_rate, ", ", green_color_rate, ", ", blue_color_rate)