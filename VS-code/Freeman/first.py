import numpy as np
import math
import time
from numpy import matrix
# Константы
h = 0.25                      # Высота датчика над полом (м)
max_traverse_height = 0.22    # Максимальная проходимая высота препятствия (м)
critical_threshold = 0.20     # Критический порог h_est (м)
min_fraction = 0.25           # Минимальная доля строк ROI с h_est >= critical_threshold 
def compute_theta(i, num_rows):
    """
    Вычисляем угол для строки i нижней половины матрицы.
    i=0 соответствует нижней строке (угол -35°), 
    i=num_rows-1 соответствует углу ближе к горизонту (0°).
    """
    theta_min = -35  # нижний угол (в градусах)
    theta_max = 0    # верхний угол
    return math.radians(theta_min + (theta_max - theta_min) * i / (num_rows - 1))
 
def analyze_ROI(matrix):
    """
    Анализируем нижнюю половину матрицы, выбираем строки с ожидаемым d_floor от 0.8 м.
    Для каждой строки вычисляем h_est = h - d_measured * sin(|θ|).
    Возвращаем медианное h_est и долю строк, где h_est >= critical_threshold.
    """
    M, N = matrix.shape
    lower_half = matrix[:M//2, :]  # нижняя половина матрицы
    num_rows = lower_half.shape[0]
    h_est_list= []
    count_high = 0
    count_total = 0
 
    for i in range(num_rows):
        theta = compute_theta(i, num_rows)
        if math.isclose(theta, 0): continue
        d_floor = h / abs(math.sin(theta))
        if 0.8 <= d_floor:
            count_total += 1
            d_measured = np.min(lower_half[i, :])
            h_est = h - d_measured * abs(math.sin(theta))
            h_est_list.append(h_est)
            if h_est >= critical_threshold:
                count_high += 1
    if h_est_list:
        median_h = np.median(h_est_list)
        fraction = count_high / count_total if count_total > 0 else 0
        return median_h, fraction
    else:
        return None, 0
 
def decide_obstacle(matrix):
    """
    Принимаем решение: если медианное h_est > max_traverse_height 
    или значительная доля строк ROI (более min_fraction) показывает h_est >= critical_threshold,
    препятствие считается опасным.
    """
    median_h, fraction = analyze_ROI(matrix)
    if median_h is None:
        print("ROI не определена, продолжаем движение")
        return False  # Нет данных – считаем, что путь свободен
    print("Медианная h_est:", median_h, "Доля строк с h_est >= critical_threshold:", fraction)
    if median_h > max_traverse_height or fraction > min_fraction:
        print("Препятствие слишком высокое, требуется объезд")
        return True
    else:                                  
        print("Путь проходим, препятствие невелико")
        return False
 
# Пример использования:
if __name__ == "__main__":
    class Robot:
        def get_depth_matrix(self,sensor_depth_name):
            # Генерация тестовой матрицы
            # В реальном приложении здесь будет код для получения данных с датчика
            return np.random.rand(16, 8) * 0.2 + 1.0  # Пример: матрица 16x8 со значениями от 1.0 до 1.2

    class Navigation:
        def change_route(self):
            # В реальном приложении здесь будет код для изменения маршрута
            print("Маршрут изменен")

    robot = Robot()
    navigation = Navigation()

    matrix = robot.get_depth_matrix("depth sensor") # Получаем матрицу измерений от датчика глубины
    print("Matrix:", matrix)
    if decide_obstacle(matrix):
        navigation.change_route() 
    else:
        # Продолжаем движение
        pass