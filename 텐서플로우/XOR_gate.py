# XOR_gate.py
# AND NAND OR XOR

def AND(x1,x2):
    w1,w2,theta = 0.5, 0.5, 0.7
    tmp = w1*x1 + w2*x2
    if tmp <= theta : # 임계값
        return 0
    elif tmp > theta :
        return 1

print(AND(0,0))  # tmp : 0
print(AND(0,1))  # tmp : 0.5
print(AND(1,0))  # tmp : 0.5
print(AND(1,1))  # tmp : 1

def NAND(x1,x2):
    w1,w2,theta = 0.5, 0.5, 0.7
    tmp = w1*x1 + w2*x2
    if tmp <= theta : # 임계값
        return 1
    elif tmp > theta :
        return 0

print(NAND(0,0))  # tmp : 0
print(NAND(0,1))  # tmp : 0.5
print(NAND(1,0))  # tmp : 0.5
print(NAND(1,1))  # tmp : 1

def OR(x1,x2):
    w1,w2,theta = 0.5, 0.5, 0.4
    tmp = w1*x1 + w2*x2
    if tmp <= theta : # 임계값
        return 0
    elif tmp > theta :
        return 1

print(OR(0,0))  # tmp : 0
print(OR(0,1))  # tmp : 0.5
print(OR(1,0))  # tmp : 0.5
print(OR(1,1))  # tmp : 1

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))


