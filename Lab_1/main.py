import random
import time

def main():
    x_array = []
    y_array = []
    a1 = 3
    a2 = 4
    a3 = 5

    for i in range(8):
        x_line = []
        for j in range(3):
            x_line.append(random.randint(1, 20))
        x_array.append(x_line)
        y_array.append(a1*x_line[0] + a2*x_line[1] + a3*x_line[2])

    x_tr_array = list(zip(x_array[0],x_array[1],x_array[2],x_array[3],x_array[4],x_array[5],x_array[6],x_array[7]))

    x_array.append([])
    x_array.append([])
    for i in range(3):
        x_array[8].append((min(x_tr_array[i]) + max(x_tr_array[i])) / 2)
        x_array[9].append(x_array[8][i] - min(x_tr_array[i]) )

    y_array.append(a1*x_array[8][0] + a2*x_array[8][1] + a3*x_array[8][2])
    y_array.append(a1*x_array[9][0] + a2*x_array[9][1] + a3*x_array[9][2])

    formating_list = list(range(8))
    formating_list.append("x0")
    formating_list.append("dx")

    for i in range(len(x_array)):
        print("{0}----a1*{1} + a2*{2} + a3*{3} = {4}".format(formating_list[i], x_array[i][0], x_array[i][1], x_array[i][2], y_array[i]))

    print("\nЗа варіантом 118 маємо критерій max(Y).\n Точка плану за критерієм:\n")

    awnser_raw_number = 0
    for i in y_array:
        if i == max(y_array):
            break
        awnser_raw_number += 1

    print("{0}----a1*{1} + a2*{2} + a3*{3} = {4}\n".format(formating_list[awnser_raw_number], x_array[awnser_raw_number][0], x_array[awnser_raw_number][1], x_array[awnser_raw_number][2],
                                                         y_array[awnser_raw_number]))


if __name__ == '__main__':
    times_arr =[]
    for times in range(100):
        start = time.time()
        main()
        stop = time.time()
        duration = stop - start
        times_arr.append(duration)
    print("\nСередній час виконання 1 ітерації:", sum(times_arr)/100)