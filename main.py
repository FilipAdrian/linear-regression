from linear_regression import *

if __name__ == '__main__':
    data_frame = read_data("./resources/apartmentComplexData.txt")
    heatmap(data_frame)
    regression, score = compute_regression(data_frame, 8)
    print(score)
    print(predict(regression, [-122, 37, 52, 3549, 707, 1551, 714, 3.691200]))
