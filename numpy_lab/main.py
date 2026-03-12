"""
Лабораторная работа: Численные вычисления и анализ данных с использованием NumPy
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Union, Any


# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================

def create_vector() -> np.ndarray:
    """
    Создать массив от 0 до 9 включительно.
    
    Returns:
        numpy.ndarray: Массив чисел от 0 до 9
    """
    return np.arange(10)


def create_matrix() -> np.ndarray:
    """
    Создать матрицу 5x5 со случайными числами в диапазоне [0, 1).
    
    Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями
    """
    return np.random.rand(5, 5)


def reshape_vector(vec: np.ndarray) -> np.ndarray:
    """
    Преобразовать вектор формы (10,) в матрицу (2, 5).
    
    Args:
        vec: Входной массив формы (10,)
    
    Returns:
        numpy.ndarray: Преобразованный массив формы (2, 5)
    
    Raises:
        ValueError: Если вектор не имеет формы (10,)
    """
    if vec.shape != (10,):
        raise ValueError(f"Ожидается вектор формы (10,), получен {vec.shape}")
    return vec.reshape(2, 5)


def transpose_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Транспонирование матрицы.
    
    Args:
        mat: Входная матрица
    
    Returns:
        numpy.ndarray: Транспонированная матрица
    """
    return mat.T


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Сложение векторов одинаковой длины.
    
    Args:
        a: Первый вектор
        b: Второй вектор
    
    Returns:
        numpy.ndarray: Результат поэлементного сложения
    
    Raises:
        ValueError: Если векторы имеют разную длину
    """
    if a.shape != b.shape:
        raise ValueError(f"Формы векторов не совпадают: {a.shape} и {b.shape}")
    return a + b


def scalar_multiply(vec: np.ndarray, scalar: Union[float, int]) -> np.ndarray:
    """
    Умножение вектора на число.
    
    Args:
        vec: Входной вектор
        scalar: Число для умножения
    
    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр
    """
    return vec * scalar


def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Поэлементное умножение двух массивов.
    
    Args:
        a: Первый массив
        b: Второй массив
    
    Returns:
        numpy.ndarray: Результат поэлементного умножения
    
    Raises:
        ValueError: Если массивы имеют разную форму
    """
    if a.shape != b.shape:
        raise ValueError(f"Формы массивов не совпадают: {a.shape} и {b.shape}")
    return a * b


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Скалярное произведение двух векторов.
    
    Args:
        a: Первый вектор
        b: Второй вектор
    
    Returns:
        float: Скалярное произведение
    
    Raises:
        ValueError: Если векторы имеют разную длину
    """
    if a.shape != b.shape:
        raise ValueError(f"Формы векторов не совпадают: {a.shape} и {b.shape}")
    return float(np.dot(a, b))


# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Умножение матриц.
    
    Args:
        a: Первая матрица
        b: Вторая матрица
    
    Returns:
        numpy.ndarray: Результат умножения матриц
    
    Raises:
        ValueError: Если матрицы нельзя перемножить
    """
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Нельзя перемножить матрицы форм {a.shape} и {b.shape}: "
            f"число столбцов первой ({a.shape[1]}) не равно числу строк второй ({b.shape[0]})"
        )
    return a @ b


def matrix_determinant(a: np.ndarray) -> float:
    """
    Вычисление определителя квадратной матрицы.
    
    Args:
        a: Квадратная матрица
    
    Returns:
        float: Определитель матрицы
    
    Raises:
        ValueError: Если матрица не квадратная
    """
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"Матрица должна быть квадратной, получена форма {a.shape}")
    return float(np.linalg.det(a))


def matrix_inverse(a: np.ndarray) -> np.ndarray:
    """
    Вычисление обратной матрицы.
    
    Args:
        a: Квадратная матрица
    
    Returns:
        numpy.ndarray: Обратная матрица
    
    Raises:
        ValueError: Если матрица вырожденная (det = 0) или не квадратная
    """
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"Матрица должна быть квадратной, получена форма {a.shape}")
    
    det = np.linalg.det(a)
    if np.abs(det) < 1e-10:
        raise ValueError("Матрица вырожденная (det ≈ 0), обратной матрицы не существует")
    
    return np.linalg.inv(a)


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решение системы линейных уравнений Ax = b.
    
    Args:
        a: Матрица коэффициентов A
        b: Вектор свободных членов b
    
    Returns:
        numpy.ndarray: Решение системы x
    
    Raises:
        ValueError: Если система не имеет единственного решения
    """
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"Матрица A должна быть квадратной, получена форма {a.shape}")
    
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Размерности A ({a.shape[0]}) и b ({b.shape[0]}) не совпадают"
        )
    
    det = np.linalg.det(a)
    if np.abs(det) < 1e-10:
        raise ValueError("Определитель матрицы близок к нулю, система не имеет единственного решения")
    
    return np.linalg.solve(a, b)


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path: str = "data/students_scores.csv") -> np.ndarray:
    """
    Загрузка данных из CSV файла.
    
    Args:
        path: Путь к CSV файлу
    
    Returns:
        numpy.ndarray: Загруженные данные
    
    Raises:
        FileNotFoundError: Если файл не найден
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл {path} не найден")
    
    return pd.read_csv(path).to_numpy()


def statistical_analysis(data: np.ndarray) -> Dict[str, float]:
    """
    Статистический анализ одномерного массива данных.
    
    Args:
        data: Одномерный массив данных
    
    Returns:
        Dict[str, float]: Словарь со статистическими показателями
    
    Raises:
        ValueError: Если массив пустой
    """
    if len(data) == 0:
        raise ValueError("Массив данных пуст")
    
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "25_percentile": float(np.percentile(data, 25)),
        "75_percentile": float(np.percentile(data, 75))
    }


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Min-Max нормализация данных в диапазон [0, 1].
    
    Формула: (x - min) / (max - min)
    
    Args:
        data: Входной массив данных
    
    Returns:
        numpy.ndarray: Нормализованный массив
    
    Raises:
        ValueError: Если массив пустой или все значения одинаковы
    """
    if len(data) == 0:
        raise ValueError("Массив данных пуст")
    
    min_val = np.min(data)
    max_val = np.max(data)
    
    if np.abs(max_val - min_val) < 1e-10:
        raise ValueError("Все значения массива одинаковы, нормализация невозможна")
    
    return (data - min_val) / (max_val - min_val)


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def ensure_plots_dir() -> None:
    """Создание директории для графиков, если её нет."""
    os.makedirs("plots", exist_ok=True)


def plot_histogram(data: np.ndarray) -> None:
    """
    Построение гистограммы распределения данных.
    
    Args:
        data: Данные для гистограммы
    """
    ensure_plots_dir()
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=10, edgecolor='black', alpha=0.7)
    plt.title('Распределение оценок по математике', fontsize=14)
    plt.xlabel('Оценка', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('plots/histogram.png', dpi=150, bbox_inches='tight')
    plt.close('all')


def plot_heatmap(matrix: np.ndarray) -> None:
    """
    Построение тепловой карты корреляции.
    
    Args:
        matrix: Матрица корреляции
    """
    ensure_plots_dir()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        square=True,
        xticklabels=['Математика', 'Физика', 'Информатика'],
        yticklabels=['Математика', 'Физика', 'Информатика']
    )
    plt.title('Тепловая карта корреляции предметов', fontsize=14)
    
    plt.savefig('plots/heatmap.png', dpi=150, bbox_inches='tight')
    plt.close('all')


def plot_line(x: np.ndarray, y: np.ndarray) -> None:
    """
    Построение графика зависимости оценок от номера студента.
    
    Args:
        x: Номера студентов
        y: Оценки студентов
    """
    ensure_plots_dir()
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'o-', linewidth=2, markersize=6, color='blue')
    plt.title('Зависимость оценки по математике от номера студента', fontsize=14)
    plt.xlabel('Номер студента', fontsize=12)
    plt.ylabel('Оценка', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(x)
    
    plt.savefig('plots/line_plot.png', dpi=150, bbox_inches='tight')
    plt.close('all')


if __name__ == "__main__":
    print("Запустите python -m pytest test.py -v для проверки лабораторной работы.")