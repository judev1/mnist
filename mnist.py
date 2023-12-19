import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.backend_bases import MouseButton


print(' :: [LOG] Importing torch')
import torch
import torch.nn as nn


image = np.zeros((28, 28))
start = []
image[0][0] = 1

def on_move(event):
    if event.button is MouseButton.LEFT:
        if event.inaxes == ax[0]:
            x = int(event.xdata + 0.5)
            y = int(event.ydata + 0.5)
            image[y][x] = 1

def on_click(event):
    global image
    if event.button is MouseButton.LEFT:
        if event.inaxes == ax[0]:
            x = int(event.xdata + 0.5)
            y = int(event.ydata + 0.5)
            image[y][x] = 1
    elif event.button is MouseButton.RIGHT:
        image = np.zeros((28, 28))


def predict():
    img = np.expand_dims(image, axis=0)
    img = torch.Tensor(img)
    preds = model(img).tolist()[0]
    for i, bar in enumerate(barchart):
        bar.set_height(preds[i])
    pred = preds.index(max(preds))
    text1.set_text(f'Maybe a {pred}?')

def updatefig(*args):
    if not start:
        image[0][0] = 0
        start.append(0)
    if not image.any():
        text1.set_text('')
        for i, bar in enumerate(barchart):
            bar.set_height(0)
    else:
        predict()
    im.set_array(image)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


print(' :: [LOG] Loading model')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('mnist.pth'))
model.eval()


print(' :: [LOG] Setting up display')
fig, ax = plt.subplots(1, 2)
fig.canvas.manager.set_window_title('Interactive MNIST model')

im = ax[0].imshow(image, cmap='gray', animated=True)
text1 = ax[0].text(0.5, 1.5, '', color='white', size=12)
im.axes.get_xaxis().set_ticks([])
im.axes.get_yaxis().set_ticks([])

values = [-4, 0, 0, 0, 0, 0, 0, 0, 0, 4]
barchart = ax[1].bar(list(map(str, range(10))), values)

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=False, cache_frame_data=False)


print(' :: [INFO] Left click to draw and right click to clear display')
plt.connect('motion_notify_event', on_move)
plt.connect('button_press_event', on_click)

plt.show()