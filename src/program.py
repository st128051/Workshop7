import math
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
import csv   # для чтения CSV

# ------------------------------------------------------------
# 1. Первая выборка (из протокола)
# ------------------------------------------------------------
U_eb1 = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37,
         0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45]
U_cb1 = [0.00260, 0.00470, 0.00650, 0.01070, 0.01690, 0.02580, 0.04280, 0.05730,
         0.07000, 0.09680, 0.11980, 0.15090, 0.18600, 0.21000, 0.22740, 0.25450]
R3 = 12.0
T = 298.15

# Вычисляем I_k и ln I_k
I_k1 = [u / R3 for u in U_cb1]
lnI_k1 = [math.log(i) for i in I_k1]

# ------------------------------------------------------------
# 2. Вторая выборка (из table_2.csv)
# ------------------------------------------------------------
# Предполагаем, что файл лежит в той же папке, что и скрипт.
# Если нет – укажи полный путь.
U_eb2 = []
U_cb2 = []
with open('table_2.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        U_eb2.append(float(row['Ueb']))
        U_cb2.append(float(row['Ukb']))

I_k2 = [u / R3 for u in U_cb2]
lnI_k2 = [math.log(i) for i in I_k2]

# ------------------------------------------------------------
# Параметры для отображения погрешностей (одинаковые для обеих выборок)
# ------------------------------------------------------------
delta_U_cb = 0.00005
delta_I_k = delta_U_cb / R3
# Относительная погрешность тока (средняя по первой выборке – для простоты используем ту же)
rel_err_avg = np.mean([delta_I_k / i for i in I_k1])
delta_lnI_k_const = rel_err_avg
VISUAL_FACTOR = 3
yerr_ln = [delta_lnI_k_const * VISUAL_FACTOR] * len(lnI_k1)
yerr_I  = [delta_I_k * VISUAL_FACTOR] * len(I_k1)

# ------------------------------------------------------------
# Функция метода парных точек
# ------------------------------------------------------------
def paired_points(x, y):
    n = len(x)
    half = n // 2
    x1 = x[:half]
    y1 = y[:half]
    x2 = x[half:]
    y2 = y[half:]
    a_i = [(y2[i] - y1[i]) / (x2[i] - x1[i]) for i in range(half)]
    a_mean = np.mean(a_i)
    a_std = np.std(a_i, ddof=1)
    t_crit = stats.t.ppf(0.975, half-1)
    delta_a = t_crit * a_std / math.sqrt(half)
    b = np.mean(y) - a_mean * np.mean(x)
    return a_mean, delta_a, b

a1, da1, b1 = paired_points(U_eb1, lnI_k1)
a2, da2, b2 = paired_points(U_eb2, lnI_k2)

print("===== Первая выборка =====")
print(f"a1 = {a1:.4f} ± {da1:.4f} (1/В), b1 = {b1:.4f}")
print(f"e/k = {T*a1:.1f} ± {T*da1:.1f} Кл/(Дж·К)")

print("\n===== Вторая выборка =====")
print(f"a2 = {a2:.4f} ± {da2:.4f} (1/В), b2 = {b2:.4f}")
print(f"e/k = {T*a2:.1f} ± {T*da2:.1f} Кл/(Дж·К)")

# ------------------------------------------------------------
# Построение графиков (обе серии)
# ------------------------------------------------------------
os.makedirs("figures", exist_ok=True)

U_fit = np.linspace(min(min(U_eb1), min(U_eb2)), max(max(U_eb1), max(U_eb2)), 100)

# График 1: ln I_k vs U_eb
plt.figure(figsize=(8,5))
plt.errorbar(U_eb1, lnI_k1, yerr=yerr_ln, fmt='o',
             markersize=3, capsize=6, ecolor='black', elinewidth=2.5,
             markeredgecolor='black', markerfacecolor='white',
             label='Выборка 1 (эксп.)')
plt.errorbar(U_eb2, lnI_k2, yerr=yerr_ln, fmt='s',   # квадратные маркеры
             markersize=3, capsize=6, ecolor='black', elinewidth=2.5,
             markeredgecolor='black', markerfacecolor='none',
             label='Выборка 2 (эксп.)')

ln_fit1 = a1 * U_fit + b1
ln_fit2 = a2 * U_fit + b2
plt.plot(U_fit, ln_fit1, 'b-', linewidth=1.5, label=f'Аппрокс. выборки 1: ln I = {a1:.3f}·U + {b1:.3f}')
plt.plot(U_fit, ln_fit2, 'g--', linewidth=1.5, label=f'Аппрокс. выборки 2: ln I = {a2:.3f}·U + {b2:.3f}')

plt.xlabel('$U_{eb}$, В')
plt.ylabel('$\\ln I_k$')
plt.title('Зависимость $\\ln I_k$ от $U_{eb}$ (две выборки)')
plt.grid(True)
plt.legend()
plt.savefig('figures/lnIk_vs_Ueb_two.png', dpi=300)
plt.show()

# График 2: I_k vs U_eb
plt.figure(figsize=(8,5))
plt.errorbar(U_eb1, I_k1, yerr=yerr_I, fmt='o',
             markersize=3, capsize=6, ecolor='black', elinewidth=2.5,
             markeredgecolor='black', markerfacecolor='white',
             label='Выборка 1 (эксп.)')
plt.errorbar(U_eb2, I_k2, yerr=yerr_I, fmt='s',
             markersize=3, capsize=6, ecolor='black', elinewidth=2.5,
             markeredgecolor='black', markerfacecolor='none',
             label='Выборка 2 (эксп.)')

I_fit1 = np.exp(a1 * U_fit + b1)
I_fit2 = np.exp(a2 * U_fit + b2)
plt.plot(U_fit, I_fit1, 'b-', linewidth=1.5, label='Экспонента (выборка 1)')
plt.plot(U_fit, I_fit2, 'g--', linewidth=1.5, label='Экспонента (выборка 2)')

plt.xlabel('$U_{eb}$, В')
plt.ylabel('$I_k$, А')
plt.title('Зависимость тока коллектора от $U_{eb}$ (две выборки)')
plt.grid(True)
plt.legend()
plt.savefig('figures/Ik_vs_Ueb_two.png', dpi=300)
plt.show()

# ------------------------------------------------------------
# Генерация LaTeX-таблиц (две отдельные таблицы)
# ------------------------------------------------------------
os.makedirs("output", exist_ok=True)

# Таблица 1 (старая)
with open("output/table1.tex", "w", encoding="utf-8") as f:
    f.write("\\begin{table}[h]\n\\centering\n")
    f.write("\\caption{Результаты измерений (первая выборка)}\n")
    f.write("\\label{tab:data1}\n")
    f.write("\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n")
    f.write("№ & $U_{eb}$, В & $U_{cb}$, В & $I_k$, А & $\\ln I_k$ \\\\\n\\hline\n")
    for i in range(len(U_eb1)):
        f.write(f"{i+1} & {U_eb1[i]:.2f} & {U_cb1[i]:.5f} & {I_k1[i]:.6f} & {lnI_k1[i]:.4f} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

# Таблица 2 (новая)
with open("output/table2.tex", "w", encoding="utf-8") as f:
    f.write("\\begin{table}[h]\n\\centering\n")
    f.write("\\caption{Результаты измерений (вторая выборка)}\n")
    f.write("\\label{tab:data2}\n")
    f.write("\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n")
    f.write("№ & $U_{eb}$, В & $U_{cb}$, В & $I_k$, А & $\\ln I_k$ \\\\\n\\hline\n")
    for i in range(len(U_eb2)):
        f.write(f"{i+1} & {U_eb2[i]:.4f} & {U_cb2[i]:.5f} & {I_k2[i]:.6f} & {lnI_k2[i]:.4f} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

print("\nТаблицы сохранены в output/, графики в figures/")

# Функция для генерации таблицы парных точек
def generate_pairs_table(x, y, suffix, caption):
    n = len(x)
    half = n // 2
    x1 = x[:half]
    y1 = y[:half]
    x2 = x[half:]
    y2 = y[half:]
    a_i = [(y2[i] - y1[i]) / (x2[i] - x1[i]) for i in range(half)]
    a_mean = np.mean(a_i)
    a_std = np.std(a_i, ddof=1)
    t_crit = stats.t.ppf(0.975, half-1)
    delta_a = t_crit * a_std / math.sqrt(half)
    with open(f"output/table_pairs{suffix}.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{tab:pairs{suffix}}}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n\\hline\n")
        f.write("№ пары & $x_{1i}$ ($U_{eb}$, В) & $y_{1i}$ ($\\ln I_k$) & $x_{2i}$ ($U_{eb}$, В) & $y_{2i}$ ($\\ln I_k$) & $a_i$ (1/В) \\\\\n\\hline\n")
        for i in range(half):
            f.write(f"{i+1} & {x1[i]:.4f} & {y1[i]:.4f} & {x2[i]:.4f} & {y2[i]:.4f} & {a_i[i]:.4f} \\\\\n")
        f.write(f"\\hline\nСреднее & & & & & {a_mean:.4f} \\\\\n")
        f.write(f"СКО & & & & & {a_std:.4f} \\\\\n")
        f.write(f"$\\Delta a$ (P=0.95) & & & & & {delta_a:.4f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

# Генерация таблиц парных точек для обеих выборок
generate_pairs_table(U_eb1, lnI_k1, "", "Расчёт методом парных точек для первой выборки")
generate_pairs_table(U_eb2, lnI_k2, "2", "Расчёт методом парных точек для второй выборки")