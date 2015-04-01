# from wikipedia example
x_old = 0
x_new = 6 # algorithm starts at x=6
eps = 0.01 # step size
precision = 0.00001

# derivative of f(x)
def f_prime(x):
	return 4 * x**3 - 9 * x**2

step = 0
print("step, x_old, x_new")
while abs(x_new - x_old) > precision:
	print(step, x_old, x_new)
	x_old = x_new
	x_new = x_old - eps * f_prime(x_old)
	step += 1

print("Local minimum occurs at ", x_new)
