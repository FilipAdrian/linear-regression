import joblib
import pprint
from linear_regression import *

file_name = "resources/price-model.pkl"
if __name__ == '__main__':
    data_frame = read_data("./resources/apartmentComplexData.txt")
    heatmap(data_frame)
    regression, score = compute_regression(data_frame, 8)
    joblib.dump(regression, file_name)
