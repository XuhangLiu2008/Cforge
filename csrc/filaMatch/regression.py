import numpy as np

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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

    @staticmethod
    def inverseGamma(x, gamma = 2.2):
        x /= 255
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** gamma

    # R_temp2coff = {4000 : 1.8}
    # G_temp2coff = {4000 : 1.0}
    # B_temp2coff = {4000 : 1.4}

    R_temp2coff = {4000 : 1.0}
    G_temp2coff = {4000 : 1.0}
    B_temp2coff = {4000 : 1.0}

    @staticmethod
    def RGB2RelativeIntensity(color, gamma = 2.2, color_temp = 4000):
        return np.array([Filament.inverseGamma(color[0]) / Filament.R_temp2coff[color_temp],
                         Filament.inverseGamma(color[1]) / Filament.G_temp2coff[color_temp],
                         Filament.inverseGamma(color[2]) / Filament.B_temp2coff[color_temp]])

    def calculateCoefficients(self, samples, shown = False, color_temp = 4000, gamma = 2.2):
        # samples is a list of [thickness, colour]
        # colour should be np.uint8 array with size 3

        def combinedCoeff(thickness, K_r, S_r, K_g, S_g, K_b, S_b, enlarge_factor_r, enlarge_factor_g, enlarge_factor_b):
            # thickness here should be an array with 3 identical copies
            # to fit all coeffs together
            length = len(thickness) // 3

            K = np.array([K_r, K_g, K_b])
            S = np.array([S_r, S_g, S_b])

            res = np.zeros((length, 3))
            for i in range(length):
                res[i], _ = Filament.RatesInAir(thickness[i], K, S, np.array([enlarge_factor_r, enlarge_factor_g, enlarge_factor_b], dtype=float))

            return res.T.flatten()

        thickness_list = []
        r_list = []
        g_list = []
        b_list = []

        for sample in samples:
            thickness_list.append(sample[0])
            intensity = Filament.RGB2RelativeIntensity(sample[1], gamma=gamma, color_temp=color_temp)
            r_list.append(intensity[0])
            g_list.append(intensity[1])
            b_list.append(intensity[2])

        reasonable_guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5]
        bounds = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [10, 10, 10, 10, 10, 10, np.inf, np.inf, np.inf])
        # constrain params to physically meaningful ranges

        MAXFEV = int(1e6)

        thickness_arr = np.asarray(thickness_list * 3, dtype=float)
        target_arr = np.asarray(r_list + g_list + b_list, dtype=float)

        coefficient, covariance = curve_fit(combinedCoeff, thickness_arr, target_arr, p0=reasonable_guess, bounds=bounds, maxfev=MAXFEV)

        r_coefficient = np.array([coefficient[0], coefficient[1], coefficient[6]])
        g_coefficient = np.array([coefficient[2], coefficient[3], coefficient[7]])
        b_coefficient = np.array([coefficient[4], coefficient[5], coefficient[8]])

        self.absorb_coeff = np.array([r_coefficient[0], g_coefficient[0], b_coefficient[0]])
        self.scatter_coeff = np.array([r_coefficient[1], g_coefficient[1], b_coefficient[1]])
        enlarge_factor = np.array([r_coefficient[2], g_coefficient[2], b_coefficient[2]])

        if self.icon_colour is None:
            tmp = (Filament.RatesInAir(100, self.absorb_coeff, self.scatter_coeff, 1))[1]
            self.icon_colour = np.uint8(tmp / np.max(tmp) * 255)

        print("R coef:", r_coefficient)
        print("G coef:", g_coefficient)
        print("B coef:", b_coefficient)
        print("absorb coeff:", self.absorb_coeff)
        print("scatter coeff:", self.scatter_coeff)
        print("icon colour:", self.icon_colour)

        if shown :
            d_sample = np.asarray(thickness_list, dtype=float)
            r_sample = np.asarray(r_list, dtype=float)
            g_sample = np.asarray(g_list, dtype=float)
            b_sample = np.asarray(b_list, dtype=float)

            d_data = np.linspace(0, np.max(d_sample) * 1.1, 1000)
            rgb_data = np.zeros((d_data.shape[0], 3), dtype=float)
            
            for i, thickness in enumerate(d_data):
                rgb_data[i], _ = Filament.RatesInAir(thickness, self.absorb_coeff, self.scatter_coeff, enlarge_factor)

            r_data = rgb_data[:, 0]
            g_data = rgb_data[:, 1]
            b_data = rgb_data[:, 2]

            plt.scatter(d_sample, r_sample, c = 'r', label='R samples')
            plt.scatter(d_sample, g_sample, c = 'g', label='G samples')
            plt.scatter(d_sample, b_sample, c = 'b', label='B samples')
            plt.plot(d_data, r_data, 'r-', label='R fit')
            plt.plot(d_data, g_data, 'g-', label='G fit')
            plt.plot(d_data, b_data, 'b-', label='B fit')
            plt.grid(color=self.icon_colour / 255)

            plt.xlabel('Thickness (mm)')
            plt.ylabel('Penetrate Rate')
            plt.legend()

            plt.show()

        return

if __name__ == '__main__':
    test_filament = Filament("test", "test")
    test_filament.calculateCoefficients([[0.1, [231, 219, 212]], [0.2, [228, 203, 153]], [0.3, [220, 180, 104]], [0.4, [219, 165, 79]], [0.5, [213, 144, 66]], [0.6, [205, 132, 57]], [0.7, [201, 119, 51]], [0.8, [201, 111, 48]], [0.9, [197, 102, 44]], [1.0, [192, 92, 40]], [1.1, [186, 86, 37]], [1.2, [184, 78, 36]], [1.3, [179, 73, 34]], [1.4, [175, 69, 33]], [1.5, [171, 65, 32]], [1.6, [164, 59, 29]]], True)