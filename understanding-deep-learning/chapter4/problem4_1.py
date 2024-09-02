# There are four piecewise linear regions going from x to y',
# and then again, there are four more linear regions going from y' to y
# Note (that I'm substituting y' <-> y as I prefer it that way).
import numpy as np
import matplotlib.pyplot as plt

# Here are the various regions

def h1(x): # from -1 to -0.4
    return x / 0.6 # y-intercept is zero

def h2(x): # from -0.4 to 0
    return 1 - ( 2 / 0.4 ) * x

def h3(x): # from 0 to 0.6
    return 2 * x / 0.6 - 1

def h4(x): # from 0.6 to 1
    return 1 - x / 0.4

def h1_prime(x): # from -1 to -0.2
    return 0.6 - 1.4 * x / (0.8)

def h2_prime(x): # from -0.2 to 0.5
    return x * 0.1 / (0.7) - 0.8

def h3_prime(x): # from 0.5 to 0.8
    return 0.7 * x / (0.3) - 0.7

def h4_prime(x): # from 0.8 to 1
    return 0

def h(x):
    if x < -0.4:
        return h1(x+1)
    elif x < 0:
        return h2(x+0.4)
    elif x < 0.6:
        return h3(x)
    else:
        return h4(x-0.6)


def hprime(x):
    if x < -0.2: 
        return h1_prime(x+1)
    elif x < 0.5:
        return h2_prime(x+0.2)
    elif x < 0.8:
        return h3_prime(x-0.5)
    else:
        return h4_prime(x-0.8)


xx = np.arange(-1,1,0.01)
yy = list()
for x in xx:
    r = h(x)
    yy.append(r)

# plot the resultant linear fitting
plt.figure(figsize=(8,6))
plt.plot(xx,yy)
plt.title('Impact of First Hidden Layer')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)

plt.savefig("original_h.png")
plt.close()

yy = list()
for x in xx:
    r = hprime(x)
    yy.append(r)

# plot the resultant linear fitting
plt.figure(figsize=(8,6))
plt.plot(xx,yy)
plt.title('Impact of Second Hidden Layer')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)

plt.savefig("original_h_prime.png")
plt.close()

# now let's apply this to the total picture.
xx = np.arange(-1,1,0.01)
yy = list()
for x in xx:
    r = h(x)
    yy.append(r) 
# we now have y', let's use to feed into h'()

result = list()
for y in yy:
    rr = hprime(y)
    result.append(rr)

plt.figure(figsize=(8,6))
plt.plot(xx,result)
plt.title('Solution to Problem 4.1')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)

plt.savefig("resultant.png")
plt.close()


