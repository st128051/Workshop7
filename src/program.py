import math
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

# ----- Данные из протокола -----
U_eb = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37,
        0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45]
U_cb = [0.00260, 0.00470, 0.00650, 0.01070, 0.01690, 0.02580, 0.04280, 0.05730,
        0.07000, 0.09680, 0.11980, 0.15090, 0.18600, 0.21000, 0.22740, 0.25450]
R3 = 12.0          # Ом
T = 298.15         # K

# Вычисляем Iк и ln Iк
I_k = [u / R3 for u in U_cb]
lnI_k = [math.log(i) for i in I_k]

# ----- Погрешности (одинаковые для всех точек) -----
delta_U_eb = 0.005      # абсолютная погрешность измерения напряжения, В (не используется в графиках, только для информации)
delta_U_cb = 0.00005    # абсолютная погрешность измерения U_cb, В
delta_I_k = delta_U_cb / R3   # абсолютная погрешность тока, А

# Вертикальные погрешности для ln I_k (через дифференциал)
delta_lnI_k = [delta_I_k / i for i in I_k]   # разные
# Чтобы сделать одинаковыми, вычислим среднюю относительную погрешность
rel_err_avg = np.mean([delta_I_k / i for i in I_k])
delta_lnI_k_const = rel_err_avg   # постоянная относительная погрешность ln I_k

# ----- Для наглядности увеличим вертикальные погрешности в 3 раза -----
VISUAL_FACTOR = 3
delta_lnI_k_const_vis = delta_lnI_k_const * VISUAL_FACTOR
delta_I_k_vis = delta_I_k * VISUAL_FACTOR

# ----- Метод парных точек -----
n = len(U_eb)
half = n // 2
x1 = U_eb[:half]
y1 = lnI_k[:half]
x2 = U_eb[half:]
y2 = lnI_k[half:]

a_i = [(y2[i] - y1[i]) / (x2[i] - x1[i]) for i in range(half)]
a_mean = np.mean(a_i)
a_std = np.std(a_i, ddof=1)
t_crit = stats.t.ppf(0.975, half-1)
delta_a = t_crit * a_std / math.sqrt(half)
b = np.mean(lnI_k) - a_mean * np.mean(U_eb)

print("===== Метод парных точек =====")
print(f"a = {a_mean:.4f} ± {delta_a:.4f} (1/В)")
print(f"b = {b:.4f}")
print(f"e/k = {T * a_mean:.1f} ± {T * delta_a:.1f} Кл/(Дж·К)")

# ----- Построение графиков без горизонтальных погрешностей -----
os.makedirs("figures", exist_ok=True)

# Вертикальные погрешности для ln I_k (увеличенные)
yerr_ln = [delta_lnI_k_const_vis] * len(lnI_k)
# Вертикальные погрешности для I_k (увеличенные)
yerr_I = [delta_I_k_vis] * len(I_k)

# График 1: ln I_k от U_eb (без горизонтальных погрешностей)
plt.figure(figsize=(8,5))
plt.errorbar(U_eb, lnI_k, yerr=yerr_ln, fmt='o',   # xerr убран
             markersize=3, markeredgewidth=0.5, capsize=6, 
             ecolor='black', elinewidth=2.5,
             markeredgecolor='black', markerfacecolor='white',
             label='Экспериментальные точки')
U_fit = np.linspace(min(U_eb), max(U_eb), 100)
ln_fit = a_mean * U_fit + b
plt.plot(U_fit, ln_fit, 'r-', linewidth=1.5, 
         label=f'Аппроксимация: ln I = {a_mean:.3f}·U + {b:.3f}')
plt.xlabel('$U_{eb}$, В')
plt.ylabel('$\\ln I_k$')
plt.title('Зависимость $\\ln I_k$ от $U_{eb}$')
plt.grid(True)
plt.legend()
plt.savefig('figures/lnIk_vs_Ueb.png', dpi=300)
plt.show()

# График 2: I_k от U_eb (без горизонтальных погрешностей)
plt.figure(figsize=(8,5))
plt.errorbar(U_eb, I_k, yerr=yerr_I, fmt='o',   # xerr убран
             markersize=3, markeredgewidth=0.5, capsize=6, 
             ecolor='black', elinewidth=2.5,
             markeredgecolor='black', markerfacecolor='white',
             label='Экспериментальные точки')
I_fit = np.exp(a_mean * U_fit + b)
plt.plot(U_fit, I_fit, 'r-', linewidth=1.5, 
         label='Экспоненциальная аппроксимация')
plt.xlabel('$U_{eb}$, В')
plt.ylabel('$I_k$, А')
plt.title('Зависимость тока коллектора от $U_{eb}$')
plt.grid(True)
plt.legend()
plt.savefig('figures/Ik_vs_Ueb.png', dpi=300)
plt.show()

# ----- Генерация LaTeX-таблиц (без упоминания погрешности по X, т.к. она не используется) -----
os.makedirs("output", exist_ok=True)

with open("output/table1.tex", "w", encoding="utf-8") as f:
    f.write("\\begin{table}[h]\n\\centering\n")
    f.write("\\caption{Результаты измерений (погрешности: $\\Delta U_{cb}=0{,}00005$ В, $\\Delta I_k=4{,}17\\cdot10^{-6}$ А, $\\Delta(\\ln I_k)\\approx 0{,}008$)}\n")
    f.write("\\label{tab:data}\n")
    f.write("\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n")
    f.write("№ & $U_{eb}$, В & $U_{cb}$, В ($\\pm0{,}00005$) & $I_k$, А ($\\pm4{,}2\\cdot10^{-6}$) & $\\ln I_k$ ($\\pm0{,}008$) \\\\\n\\hline\n")
    for i in range(n):
        f.write(f"{i+1} & {U_eb[i]:.2f} & {U_cb[i]:.5f} & {I_k[i]:.6f} & {lnI_k[i]:.4f} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

with open("output/table_pairs.tex", "w", encoding="utf-8") as f:
    f.write("\\begin{table}[h]\n\\centering\n")
    f.write("\\caption{Расчёт методом парных точек}\n")
    f.write("\\label{tab:pairs}\n")
    f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n\\hline\n")
    f.write("№ пары & $x_{1i}$ ($U_{eb}$, В) & $y_{1i}$ ($\\ln I_k$) & $x_{2i}$ ($U_{eb}$, В) & $y_{2i}$ ($\\ln I_k$) & $a_i$ (1/В) \\\\\n\\hline\n")
    for i in range(half):
        ai = (y2[i] - y1[i]) / (x2[i] - x1[i])
        f.write(f"{i+1} & {x1[i]:.2f} & {y1[i]:.4f} & {x2[i]:.2f} & {y2[i]:.4f} & {ai:.4f} \\\\\n")
    f.write(f"\\hline\nСреднее & & & & & {a_mean:.4f} \\\\\n")
    f.write(f"СКО & & & & & {a_std:.4f} \\\\\n")
    f.write(f"$\\Delta a$ (P=0.95) & & & & & {delta_a:.4f} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

print("\nТаблицы сохранены в output/, графики в figures/")
print("Горизонтальные погрешности убраны, вертикальные увеличены в 3 раза для наглядности.")