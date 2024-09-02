import numpy as np
import matplotlib.pyplot as plt
import pdb

# overall bias
bias = 2.5
# weights for the hidden functions
wh_1 = 0.7
wh_2 = 0.45
wh_3 = 0.8

def hidden_one(x):
    h = -18.5 - 4.5*x
    return h

def hidden_two(x):
    h = 16 + 2*x
    return h

def hidden_three(x):
    h = 9.0 - 3*x
    return h

def activation( hidden, x ):
    relu = hidden(x)
    if relu < 0:
        return 0
    else:
        return relu 

xx = np.arange(-10,10,0.1)
activated = list()
yy = list()

for x in xx:
    h1 = activation( hidden_one, x )
    h2 = activation( hidden_two, x )
    h3 = activation( hidden_three, x)

    label = ''
    if h1 > 0: 
        label += 'h1-'
    if h2 > 0:
        label += 'h2-'
    if h3 > 0:
        label += 'h3'
    
    last_label = ''
    if len(activated):
        last_label = activated[-1].split(' ')[-1]

    if last_label == '' or last_label != label:
        activated.append( f"{x} {label}" )
    
    y = bias + wh_1 * h1 + wh_2 + h2 + wh_3 + h3 
    yy.append(y) 

# dump the activations to a file to check
with open('activations.dat','w') as file:
    for item in activated:
        file.write(f"{item}\n")

# plot the resultant linear fitting
plt.figure(figsize=(8,6))
plt.plot(xx,yy)
plt.title('Impact of hidden units in Shallow Neural Network')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)

plt.savefig("resultant.png")
plt.close()