import random
import numpy as np
from numpy.linalg import solve
from scipy.stats import f, t
import time



def main(n):
    # варіант 118
    x1min = 20
    x1max = 70
    x2min = 25
    x2max = 65
    x3min = 25
    x3max = 35

    # максимальне та мінімальне значення
    y_max = 200 + (x1max + x2max + x3max) / 3
    y_min = 200 + (x1min + x2min + x3min) / 3

    # матриця ПФЕ
    xn = [[1, 1, 1, 1, 1, 1, 1, 1],
          [-1, -1, 1, 1, -1, -1, 1, 1],
          [-1, 1, -1, 1, -1, 1, -1, 1],
          [-1, 1, 1, -1, 1, -1, -1, 1]]

    x1x2_norm, x1x3_norm, x2x3_norm, x1x2x3_norm = [0] * 8, [0] * 8, [0] * 8, [0] * 8

    for i in range(n):
        x1x2_norm[i] = xn[1][i] * xn[2][i]
        x1x3_norm[i] = xn[1][i] * xn[3][i]
        x2x3_norm[i] = xn[2][i] * xn[3][i]
        x1x2x3_norm[i] = xn[1][i] * xn[2][i] * xn[3][i]

    # заповнення у(генерація)
    y1 = [random.randint(int(y_min), int(y_max)) for i in range(8)]
    y2 = [random.randint(int(y_min), int(y_max)) for i in range(8)]
    y3 = [random.randint(int(y_min), int(y_max)) for i in range(8)]

    # матриця планування
    y_matrix = [[y1[0], y2[0], y3[0]],
                [y1[1], y2[1], y3[1]],
                [y1[2], y2[2], y3[2]],
                [y1[3], y2[3], y3[3]],
                [y1[4], y2[4], y3[4]],
                [y1[5], y2[5], y3[5]],
                [y1[6], y2[6], y3[6]],
                [y1[7], y2[7], y3[7]]]

    # вивід данних за допомогою цикла
    print("Матриця планування y : \n")
    for i in range(n):
        print(y_matrix[i])

    x0 = [1, 1, 1, 1, 1, 1, 1, 1]

    # заміна -1 на х1_мін, 1 на х1_макс
    x1 = [x1min, x1min, x1max, x1max, x1min, x1min, x1max, x1max]

    # заміна -1 на х2_мін, 1 на х2_макс
    x2 = [x2min, x2max, x2min, x2max, x2min, x2max, x2min, x2max]

    # заміна -1 на х3_мін, 1 на х3_макс
    x3 = [x3min, x3max, x3max, x3min, x3max, x3min, x3min, x3max]

    # заповнення нулями х1х2, х1х3, х1х2х3
    x1x2, x1x3, x2x3, x1x2x3 = [0] * 8, [0] * 8, [0] * 8, [0] * 8
    # заповнення х1х2, х1х3, х1х2х3 добутками
    for i in range(n):
        x1x2[i] = x1[i] * x2[i]
        x1x3[i] = x1[i] * x3[i]
        x2x3[i] = x2[i] * x3[i]
        x1x2x3[i] = x1[i] * x2[i] * x3[i]
    # середні у
    koh_begin = time.clock()
    Y_average = []
    for i in range(len(y_matrix)):
        Y_average.append(np.mean(y_matrix[i], axis=0))

    # формуємо списки b i a
    list_for_b = [xn[0], xn[1], xn[2], xn[3], x1x2_norm, x1x3_norm, x2x3_norm, x1x2x3_norm]
    list_for_a = list(zip(x0, x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3))

    # вивід матриці планування Х
    print("Матриця планування X:")
    for i in range(n):
        print(list_for_a[i])
    # нормовані фактори b_i
    bi = []
    for k in range(n):
        S = 0
        for i in range(n):
            S += (list_for_b[k][i] * Y_average[i]) / n
        bi.append(round(S, 3))
    # розрахунок аі(система рівнянь) через функцію solve, вивід рівняння регресії
    ai = [round(i, 3) for i in solve(list_for_a, Y_average)]
    print("Рівняння регресії: \n" "y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3".format(ai[0],
                                                                                                                    ai[1],
                                                                                                                    ai[2],
                                                                                                                    ai[3],
                                                                                                                    ai[4],
                                                                                                                    ai[5],
                                                                                                                    ai[6],
                                                                                                                    ai[7]))
    # вивід даних
    print("Час виконання перевірки за критерієм Кохрена: " + str(time.clock() - koh_begin))
    print("Рівняння регресії для нормованих факторів: \n" "y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 +"
          " {}*x2x3 + {}*x1x2x3".format(bi[0], bi[1], bi[2], bi[3], bi[4], bi[5], bi[6], bi[7]))

    print("Перевірка за критерієм Кохрена")
    print("Середні значення відгуку за рядками:", "\n", +Y_average[0], Y_average[1], Y_average[2], Y_average[3],
          Y_average[4], Y_average[5], Y_average[6], Y_average[7])
    # розрахунок дисперсій
    dispersions = []
    for i in range(len(y_matrix)):
        a = 0
        for k in y_matrix[i]:
            a += (k - np.mean(y_matrix[i], axis=0)) ** 2
        dispersions.append(a / len(y_matrix[i]))
    # експериментально
    Gp = max(dispersions) / sum(dispersions)
    # теоретично
    Gt = 0.5157
    # перевірка однорідності дисперсій

    if Gp < Gt:
        print("Дисперсія однорідна")
    else:
        print("\nДисперсія неоднорідна, потрібно розпочати експеримет з початку\n")
        main(n + 1)

    # критерій Стьюдента
    student_begin = time.clock()
    print(" Перевірка значущості коефіцієнтів за критерієм Стьюдента")
    sb = sum(dispersions) / len(dispersions)
    sbs = (sb / (8 * 3)) ** 0.5

    t_list = [abs(bi[i]) / sbs for i in range(0, 8)]

    d = 0
    res = [0] * 8
    coef_1 = []
    coef_2 = []
    # кількість повторень кожної комбінації
    m = 3
    F3 = (m - 1) * n
    # перевірка значущості коефіцієнтів
    for i in range(n):
        if t_list[i] < t.ppf(q=0.975, df=F3):
            coef_2.append(bi[i])
            res[i] = 0
        else:
            coef_1.append(bi[i])
            res[i] = bi[i]
            d += 1

    # вивід
    print("Час виконання перевірки за критерієм Стьоюдента: "+ str(time.clock()-student_begin))
    print("Значущі коефіцієнти регресії:", coef_1)
    print("Незначущі коефіцієнти регресії:", coef_2)

    # значення y з коефіцієнтами регресії
    y_st = []
    for i in range(n):
        y_st.append(res[0] + res[1] * xn[1][i] + res[2] * xn[2][i] + res[3] * xn[3][i] + res[4] * x1x2_norm[i] \
                    + res[5] * x1x3_norm[i] + res[6] * x2x3_norm[i] + res[7] * x1x2x3_norm[i])
    print("Значення з отриманими коефіцієнтами:\n", y_st)

    # критерій Фішера
    phisher_begin = time.clock()
    print("\nПеревірка адекватності за критерієм Фішера\n")
    Sad = m * sum([(y_st[i] - Y_average[i]) ** 2 for i in range(8)]) / (n - d)
    Fp = Sad / sb
    F4 = n - d
    if Fp < f.ppf(q=0.95, dfn=F4, dfd=F3):
        print("Рівняння регресії адекватне при рівні значимості 0.05")
    else:
        print("Рівняння регресії неадекватне при рівні значимості 0.05")
    print("Час виконання перевірки за критерієм Фішера: " + str(time.clock() - phisher_begin))

main(8)