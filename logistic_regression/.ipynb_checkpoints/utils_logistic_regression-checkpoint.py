import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_house_data():
    house_data = np.loadtxt('data/housing_price_labelled.txt', skiprows=1, delimiter=',')
    sizes = house_data[:,0]
    prices = house_data[:,1]
    labels = house_data[:,2].astype(int)
    return sizes, prices, labels


def plot_house_data(sizes, prices, labels, scaled=False):
    
    fig = plt.figure(figsize=(6,4))
    axes = fig.add_subplot(1,1,1)
    
# Condition: Labels = 0 Apartments
    mask = labels == 0 

    axes.scatter(sizes[mask], prices[mask], color='blue', label='Apartments')
    axes.scatter(sizes[~mask], prices[~mask], color='red', label='House')
    
    if scaled:
        axes.set_xlabel('Size [a.u.]')
        axes.set_ylabel('Price [a.u.]')
               
    else:
        axes.set_xlabel('Size [sqm]')
        axes.set_ylabel('Price [CHF]')
        
    axes.grid()
    axes.legend()
    
def animate_logistic_gradient_descent(features, labels, w_path, b_path, loss_path, feat1, feat2):
    fig = plt.figure(figsize=(10,4))
      
      # Plot 1: Decision boundary in feature space
    ax1 = fig.add_subplot(1,2,1)
    
    mask = labels == 0 
    size_data = features[:,0]
    price_data = features[:,1]
    
    ax1.scatter(size_data[mask], price_data[mask], color='blue', label='Apartments')
    ax1.scatter(size_data[~mask], price_data[~mask], color='red', label='House')
    
    ax1.set_xlabel('Size [a.u.]')
    ax1.set_ylabel('Price [a.u.]')
    
    line1, = ax1.plot([],[],'b-',linewidth=2)
    
    feat1_data = features[:, feat1]
    feat2_data = features[:, feat2]

    x_line = np.linspace(feat1_data.min(), feat1_data.max(),100)
  
    ax1.grid()
    ax1.legend()
    
     # Plot 2: Loss History
    feat1_data = features[:,feat1]


    def animate(frame):
        
        step = min(frame, len(w_path)-1)
        
        current_w = w_path[step]
        current_b = b_path[step]

        y_line = -(current_w[feat1] * x_line + current_b) / current_w[feat2]
        line1.set_data(x_line, y_line)
        return line1#, text_ax1
        

    n_frames = len(w_path)
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=400, repeat=True)
    return anim
    
