import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#read all the .png files in directory called `steps`
files = glob.glob("output/frames_static_obs/*.png")
files.sort()

from PIL import Image

image_array = []

for my_file in files:
    
    image = Image.open(my_file)
    image_array.append(image)

print('image_arrays shape:', np.array(image_array).shape)

# Create the figure and axes objects
fig, ax = plt.subplots()

ax.set_axis_off()

# Set the initial image
im = ax.imshow(image_array[0], animated=True)

def update(i):
    im.set_array(image_array[i])
    return im, 

# Create the animation object
animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=200, blit=True,repeat_delay=10,)

# Show the animation
plt.show()

animation_fig.save("output/GIFs/static_obs.gif")