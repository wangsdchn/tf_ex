def line(a,b):
    def point(x):
        print(a*x+b)
        return a*x+b
    return point

line1 = line(1,1)
line2 = line(2,2)
line1(1)
line2(1)
line1(2)
line2(2)
