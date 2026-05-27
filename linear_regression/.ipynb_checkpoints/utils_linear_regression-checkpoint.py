import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_house_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    sizes = data[:,0]
    prices = data[:,1]
    labels = data[:,2].astype(int) # 0 = Apartment, 1 = house
    return sizes, prices, labels

def plot_housing_prices(sizes,prices,scaled=False):
    fig, axes = plt.subplots(figsize=(6,4))
    axes.scatter(sizes,prices,label='Data points',color="blue")
    axes.set_title('Housing Price vs Sizes')
    if scaled:
        axes.set_xlabel('Size [a.u.]')
        axes.set_ylabel('Price [a.u.]')
               
    else:
        axes.set_xlabel('Size [sqm]')
        axes.set_ylabel('Price [CHF]')
        
    axes.grid()
    axes.legend()

def plot_loss_landscape(W, B, Loss):
    fig = plt.figure(figsize=(10,8))
    axes = fig.add_subplot(111, projection='3d')
    surface = axes.plot_surface(W, B, Loss, linewidth=0, alpha = 0.7, cmap = 'viridis')
    axes.set_title('Loss Landscape')
    axes.set_xlabel('Weight (w)')
    axes.set_ylabel('Bias (b)')
    axes.set_zlabel('Loss (MSE)')
    
    fig.colorbar(surface, shrink=0.5, aspect=5)



def animate_gradient_descent(W, B, Loss, x, y, w_path, b_path, loss_path):
    
    fig = plt.figure(figsize=(20,6))
    ax1 = fig.add_subplot(1,3,1)
    ax1.scatter(x,y,label='Data points',color="blue")
    
    ax1.set_title('Housing Price vs Sizes')
    ax1.set_xlabel('Size [a.u.]')
    ax1.set_ylabel('Price [a.u.]')
    ax1.set_xlabel('Size [sqm]')
    ax1.set_ylabel('Price [CHF]')    
    ax1.grid()
    ax1.legend()
    
    x_line = np.linspace(x.min(),x.max())
    line1, = ax1.plot([],[],'r-',linewidth=2)
    
    ax2 = fig.add_subplot(1,3,2, projection='3d')
    surface = ax2.plot_surface(W, B, Loss, linewidth=0, alpha = 0.7, cmap = 'viridis')
    ax2.set_title('Loss Landscape')
    ax2.set_xlabel('Weight (w)')
    ax2.set_ylabel('Bias (b)')
    ax2.set_zlabel('Loss (MSE)')    
    ax2.view_init(elev=25, azim=-70)
    path_line, = ax2.plot([],[],[],'r-o',linewidth=2,markersize=4)
    
    
    
    ax3 = fig.add_subplot(1,3,3)
    
    #ax3.scatter(loss_path,np.arange(loss_path.size), label='Loss',s=10)
    
    
    loss_line, = ax3.plot([], [], 'b-o', linewidth=2, markersize=4)
    ax3.set_title('Loss convergence during training')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss [MSE]')    
    ax3.grid()
    ax3.set_xlim(0, len(loss_path)-1)
    ax3.set_ylim(min(loss_path) * 0.95, max(loss_path) * 1.05)
    
    
    plt.tight_layout()
    
    text_ax1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    
    text_ax3 = ax3.text(0.02, 0.98, '', transform=ax3.transAxes, fontsize=10,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
        


    def animate(frame):
 
        step = min(frame, len(w_path)-1)
        #print(step,frame)
        #print(step,loss_path[step],len(loss_path))
        current_w = w_path[step]
        current_b = b_path[step]
        y_line = current_w * x_line + current_b
        line1.set_data(x_line, y_line)
        text_ax1.set_text(f"Step: {frame}\nw: {current_w:.4f}\nb: {current_b:.4f}")


    #Update path on Loss Landscape
        path_line.set_data_3d(w_path[:step+1], b_path[:step + 1], loss_path[:step + 1])
   
    # Update loss plot
        steps_so_far = np.arange(step + 1)
        loss_so_far = loss_path[:step + 1]
        loss_line.set_data(steps_so_far,loss_so_far)
        text_ax3.set_text(f"Step: {frame}\nLoss: {loss_path[step]:.4f}")
    
        return line1, text_ax1, path_line, loss_line, text_ax3

    n_frames = len(w_path)

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=400, repeat=True)


    return anim
