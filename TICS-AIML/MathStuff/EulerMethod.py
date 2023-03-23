
def func(x, y):
    return 2*y - 4*x


def euler(x0, y, h, xn):
    while (x0 < xn):
        y = y + h * func(x0, y)
        x0 += h
        print(f"({x0}, {y})")
    print("Approximation: x: " + str(xn) + ", y: " + str(y))


euler(2, 6, 0.2, 3)
