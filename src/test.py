
def f(p=[]):
    p.append(1)
    return p

def g(p=[]):
    p.append(1)
    return p

a = f()
print(a)
b = g()
print(b)
c = f()
print(a, b, c)