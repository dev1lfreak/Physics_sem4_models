import numpy as np
import matplotlib.pyplot as plt

L = float(input("L, м: "))
d = float(input("d, м: "))
D = float(input("D, м: "))
U = float(input("U, В: "))
rho = float(input("rho, Ом·м: "))
mu0 = 4 * np.pi * 1e-7


N = L / (np.pi * D)
l_min = N * d
print(f"\nN = {N:.2f}")
print(f"l_min = {l_min:.4f} м")

l_max = float(input(f"\nl_max >= {l_min:.4f} м: "))
num_points = 2000

if l_max < l_min:
    print("Ошибка: l_max должно быть >= l_min. Установлено l_max = l_min.")
    l_max = l_min


l_values = np.linspace(l_min, l_max, num_points)
B_values = (mu0 * U * (d**2)) / (4 * rho * D * l_values)

L_ind_values = (mu0 * (L**2)) / (4 * np.pi * l_values)

plt.figure(figsize=(10, 6))
plt.plot(l_values, B_values, label=f'Функция зависимости B(l)')
plt.xlabel('Длина катушки l, м')
plt.ylabel('Магнитная индукция B, Тл')
plt.title('Зависимость магнитной индукции от длины катушки')
plt.grid(True)
plt.legend()
plt.show()


print(f"\nl = {l_min:.4f} м: {L_ind_values[0]:.4e} Гн")