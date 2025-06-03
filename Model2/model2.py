import numpy as np
import matplotlib.pyplot as plt

# 1) Функции для спектра и конвертации цвета

def wavelength_to_rgb(wl_nm):
    """
    Аппроксимация перевода длины волны (nm) в RGB.
    Возвращает кортеж (R, G, B), каждое от 0 до 1.
    """
    wl = wl_nm
    if wl >= 380 and wl < 440:
        R = -(wl - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif wl >= 440 and wl < 490:
        R = 0.0
        G = (wl - 440) / (490 - 440)
        B = 1.0
    elif wl >= 490 and wl < 510:
        R = 0.0
        G = 1.0
        B = -(wl - 510) / (510 - 490)
    elif wl >= 510 and wl < 580:
        R = (wl - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif wl >= 580 and wl < 645:
        R = 1.0
        G = -(wl - 645) / (645 - 580)
        B = 0.0
    elif wl >= 645 and wl <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0.0

    # Корректировка яркости ближе к границам диапазона
    if wl >= 380 and wl < 420:
        factor = 0.3 + 0.7*(wl - 380) / (420 - 380)
    elif wl >= 420 and wl < 701:
        factor = 1.0
    elif wl >= 701 and wl <= 780:
        factor = 0.3 + 0.7*(780 - wl) / (780 - 700)
    else:
        factor = 0.0

    return (R*factor, G*factor, B*factor)


# 2) Функции для расчёта интенсивности

def thickness(r, R):
    """
    Толщина воздушного слоя h(r) = r^2/(2R).
    r, R в метрах.
    """
    return r**2 / (2 * R)

def phase_diff(r, wavelength_m, R):
    """
    Фазовый сдвиг в точке r (радианы):
    wavelength_m, r и R в метрах.
    """
    return (2 * np.pi * r**2) / (wavelength_m * R)

def intensity_mono(r, wavelength_m, R):
    """
    Интенсивность монохроматического в каждой точке r (ненормированная, от 0 до 2I0).
    Можем считать I0=1.
    """
    phi = phase_diff(r, wavelength_m, R)
    return 2.0 * (np.sin(phi/2))**2


def intensity_quasi(r_grid, lambda_array, R):
    """
    Интенсивность квазимонохроматического:
    r_grid - двумерный массив радиусов [m x m],
    lambda_array - одномерный массив длин волн (м).
    """
    ny, nx = r_grid.shape
    N_lambda = len(lambda_array)
    # Инициализируем массивы
    I_sum = np.zeros_like(r_grid)         # сумма интенсивностей по λ
    R_sum = np.zeros_like(r_grid)
    G_sum = np.zeros_like(r_grid)
    B_sum = np.zeros_like(r_grid)

    # Пройдёмся по всем λ
    for wl in lambda_array:
        I_tmp = intensity_mono(r_grid, wl, R)  # интенсивность для данной λ
        # цвет для данной λ
        clr = wavelength_to_rgb(wl*1e9)   # переводим в нм для функции
        R_sum += clr[0] * I_tmp
        G_sum += clr[1] * I_tmp
        B_sum += clr[2] * I_tmp
        I_sum += I_tmp

    # Чтобы получить цвета, делим сумму RGB·I на сумму I (нормируем яркость)
    # Добавим малый eps, чтобы не делить на ноль.
    eps = 1e-12
    R_img = R_sum / (I_sum + eps)
    G_img = G_sum / (I_sum + eps)
    B_img = B_sum / (I_sum + eps)

    # Нормируем итоговую интенсивность I_sum, чтобы самой яркости в цветном изображении не "заваливало":
    I_avg = I_sum / N_lambda  # можно считать, что I0=1, поэтому нормировка внутри 0..2
    I_norm = I_avg / np.max(I_avg + eps)

    # Итоговое RGB-изображение (тёмные участки «чёрные», светлые — цветные).
    RGB_image = np.stack((R_img*I_norm, G_img*I_norm, B_img*I_norm), axis=-1)
    return I_avg, RGB_image


# 3) Основная функция: построение изображений и графиков

def simulate_newton_rings():
    # --- Ввод данных ---
    print("### Моделирование колец Ньютона ###\n___________________________________")
    R = float(input("Радиус кривизны линзы R (в метрах): "))   # например, 1.0
    print("Выберите режим:\n  1 — монохроматический\n  2 — квазимонохроматический")
    mode = int(input("Ваш выбор (1 или 2): "))
    if mode == 1:
        lam_nm = float(input("Длина волны λ (нм): "))
        lam = lam_nm * 1e-9  # переводим в метры
        lambda_array = np.array([lam])
    elif mode == 2:
        lam0_nm = float(input("Центральная длина волны λ₀ (нм): "))
        dlam_nm = float(input("Ширина спектра Δλ (нм): "))
        N_lambda = int(input("Число точек дискретизации по λ (например, 50–200): "))
        lam_min = (lam0_nm - dlam_nm/2) * 1e-9
        lam_max = (lam0_nm + dlam_nm/2) * 1e-9
        lambda_array = np.linspace(lam_min, lam_max, N_lambda)
    else:
        print("Неправильный режим. Завершение.")
        return

    # Параметры двумерной сетки
    img_size = int(input("Размер выходного изображения (число пикселей по ширине, квадратное): "))
    r_max_mm = float(input("Максимальный радиус r_max (в мм) для визуализации: "))
    # переводим r_max в метры и задаём координатную сетку
    r_max = r_max_mm * 1e-3
    # создаём равномерную сетку x,y = [−r_max, r_max]
    x = np.linspace(-r_max, r_max, img_size)
    y = np.linspace(-r_max, r_max, img_size)
    xx, yy = np.meshgrid(x, y)
    r_grid = np.sqrt(xx**2 + yy**2)

    # Рассчёт интенсивностей
    if mode == 1:
        # Монохроматический свет
        I_mono = intensity_mono(r_grid, lambda_array[0], R)
        # Нормировка для отображения (0..1)
        I_mono_norm = I_mono / np.max(I_mono)

        # Отображение интерференционной картины
        rgb_color = wavelength_to_rgb(lam_nm)

        # Создаем цветное изображение, масштабируя интенсивность по каждому каналу
        img_rgb = np.zeros((img_size, img_size, 3))
        for i in range(3):
            img_rgb[..., i] = I_mono_norm * rgb_color[i]

        # Отображаем цветную интерференционную картину
        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb, extent=(-r_max_mm, r_max_mm, -r_max_mm, r_max_mm))
        plt.title(f"Цветные кольца Ньютона, λ={lam_nm:.1f} нм, R={R} м")
        plt.xlabel("x (мм)")
        plt.ylabel("y (мм)")
        plt.tight_layout()
        plt.show()

        # Построим график I(r) при r вдоль горизонтальной оси
        center_row = img_size // 2
        I_cut = I_mono_norm[center_row, :]
        plt.figure(figsize=(7,4))
        plt.plot(x*1e3, I_cut, color='black')
        plt.title("Интенсивность вдоль радиуса (монохроматический)")
        plt.xlabel("r (мм)"); plt.ylabel("Нормированная I")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        # Квазимонохроматический свет
        I_avg, RGB_img = intensity_quasi(r_grid, lambda_array, R)

        # Отображаем цветную интерференционную картину
        plt.figure(figsize=(6,6))
        mask = r_grid <= r_grid
        canvas = np.zeros_like(RGB_img)
        canvas[mask] = RGB_img[mask]
        plt.imshow(canvas, extent=(-r_max_mm, r_max_mm, -r_max_mm, r_max_mm))
        plt.title(f"Квазимонохроматические кольца, λ₀={lam0_nm:.1f}±{dlam_nm/2:.1f} нм")
        plt.xlabel("x (мм)"); plt.ylabel("y (мм)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # График усреднённой интенсивности I(r)
        I_avg_norm = I_avg / np.max(I_avg + 1e-12)
        center_row = img_size // 2
        I_cut_q = I_avg_norm[center_row, :]
        plt.figure(figsize=(7,4))
        plt.plot(x*1e3, I_cut_q, color='black')
        plt.title("Интенсивность вдоль радиуса (квазимонохроматический)")
        plt.xlabel("r (мм)"); plt.ylabel("Нормированная I")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# 4) Точка входа

if __name__ == "__main__":
    simulate_newton_rings()
