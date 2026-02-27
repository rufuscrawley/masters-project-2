# Normalisation variables for the spreadsheets
# Key values are [(Logarithmic normalisation), (Invert normalisation)]
import keras

names = {
    "amin1": {
        "logarithmic": True,
        "invert": True
    },
    "amax1": {
        "logarithmic": True,
        "invert": False
    },
    "inclinations": {
        "logarithmic": False,
        "invert": False
    },
    "Stellar_age": {
        "logarithmic": True,
        "invert": False
    },
    "mass1": {
        "logarithmic": False,
        "invert": False
    },
    "Temp_sublimation": {
        "logarithmic": True,
        "invert": False
    },
    "router": {
        "logarithmic": False,
        "invert": False
    },
    "height": {
        "logarithmic": False,
        "invert": False
    },
    "betadisc": {
        "logarithmic": False,
        "invert": False
    },
    "alphadisc": {
        "logarithmic": False,
        "invert": False
    },
    "mdisc": {
        "logarithmic": True,
        "invert": True
    },
    "Stellar_radius": {
        "logarithmic": False,
        "invert": False
    },
    "Stellar_temperature": {
        "logarithmic": False,
        "invert": False
    },
    "rinner": {
        "logarithmic": False,
        "invert": False
    },
}
wavelengths = [0.1, 0.11, 0.121, 0.133, 0.147, 0.161, 0.178, 0.195, 0.215, 0.237, 0.26, 0.286, 0.315, 0.347, 0.382,
               0.42, 0.462,
               0.509, 0.56, 0.616, 0.678, 0.746, 0.821, 0.903, 0.994, 1.09, 1.2, 1.32, 1.46, 1.6, 1.76, 1.94, 2.14,
               2.35, 2.59,
               2.85, 3.13, 3.45, 3.79, 4.17, 4.59, 5.06, 5.56, 6.12, 6.74, 7.41, 8.16, 8.98, 9.88, 10.9, 12, 13.2, 14.5,
               15.9,
               17.5, 19.3, 21.2, 23.4, 25.7, 28.3, 31.1, 34.3, 37.7, 41.5, 45.7, 50.2, 55.3, 60.8, 66.9, 73.7, 81.1,
               89.2, 98.2,
               108, 119, 131, 144, 158, 174, 192, 211, 232, 256, 281, 309, 341, 375, 412, 454, 499, 549, 605, 665, 732,
               806, 887,
               976, 1070, 1180, 1300]
# GLOBAL VARIABLES
filename = "outputs"
file = f"datasets/{filename}.csv"
n_file = f'datasets/normalised/n_{filename}.csv'
const_file = f'datasets/constants/const_{filename}.csv'
excluded = ["ninc",
            "Stellar_age",
            "mass1",
            "Temp_sublimation",
            "alphadisc",
            "rinner"]
split = 15 - len(excluded)
# model = keras.models.load_model(f"models/{filename}_model.keras")
