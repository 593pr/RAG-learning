def c_to_f(C):
    F = C * 1.8 + 32
    return F
def f_to_c(F):
    C = (F - 32)/1.8
    return C
def batch_convert(t,direction):
    result = []
    if direction =="to_f":                                      
        for temp in t:
            result.append(c_to_f(temp))
    if direction =="to_C":
        for temp in t:
            result.append(f_to_c(temp))
    return result


print(c_to_f(0))                         
print(f_to_c(100))                       
print(batch_convert([0, 20, 100], "to_f"))