def harris_benedict(w, h, a, s):
    return 655.1 + 9.563*w + 1.85*h - 4.676*a if s == 'f' else 66.5 + 13.75*w + 5.003*h - 6.775*a

w = 87
h = (7, 5) # feet, inches
h = h[0] * 30.48 + h[1] * 2.54 # cm
a = 60

def bmi(w, h, a):
    return w / (h/100)**2