import pathlib
import os

import numpy as np
import plotly.graph_objs as go


def load_data_from_folder(folder_path):
    data = []
    labels = []

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            class_name = os.path.splitext(filename)[0]  # Extract class name from file name
            labels.append(class_name)
            d = np.loadtxt(file_path, delimiter=",")
            data.append(d)

    return data, labels


class Data:
    def __init__(self, path: pathlib.Path):
        # data shape [n, 2] first col is res_time, second is values
        self._temp = None
        self._light = None
        self._res_time = None
        self._cat_conc = None
    
        self._ir = None
        self._nmr = None
        self._Mn = None
        self._D = None

        self.load(path)

    def load(self, path: pathlib.Path):
        data, labels = load_data_from_folder(path)
        for d, l in zip(data, labels):
            if not hasattr(self, f"_{l}"):
                raise ValueError(f"class doesn't have '{l}'")
            setattr(self, f"_{l}", d)

    def temp(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self._temp[:, 0], self._temp[:, 1]) -273

    def light(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self._light[:, 0], self._light[:, 1])
    
    def res_time(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self._res_time[:, 0], self._res_time[:, 1])
    
    def cat_conc(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self._cat_conc[:, 0], self._cat_conc[:, 1])
    
    def ir(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self._ir[:, 0], self._ir[:, 1])
    
    def Mn(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self._Mn[:, 0], self._Mn[:, 1])
    
    def D(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self._D[:, 0], self._D[:, 1])
    
    def nmr(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self._nmr[:, 0], self._nmr[:, 1])


# def scatter_matrix_plot(data: Data):
#     x = np.linspace(0, 444*60, 1000)
#
#     fig = go.Figure(data=go.Splom(
#                 dimensions=[
#                     dict(label='temperature', values=data.temp(x)),
#                     dict(label='light', values=data.light(x)),
#                     dict(label='time', values=data.res_time(x)),
#                     dict(label='cat. conc', values=data.cat_conc(x)),
#                     dict(label='conversion', values=data.ir(x)),
#                     dict(label='Mn', values=data.Mn(x)),
#                     dict(label='D', values=data.D(x)),
#                 ],
#                 showupperhalf=False,  # remove plots on diagonal
#                 # marker=dict(color=index_vals,
#                 #             showscale=False, # colors encode categorical variables
#                 #             line_color='white', line_width=0.5)
#     ))
#
#     fig.update_layout(
#         width=1200,
#         height=1200,
#     )
#
#     fig.show()

def scatter_matrix_plot(data):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import pandas as pd
    x = np.linspace(0, 444*60, 1000)
    df = pd.DataFrame({
        'temperature': data.temp(x),
        'light': data.light(x),
        'time': data.res_time(x),
        'cat. conc': data.cat_conc(x),
        'conversion': data.ir(x),
        'Mn': data.Mn(x),
        'D': data.D(x)
    })



    axes = pd.plotting.scatter_matrix(df, figsize=(12, 12), diagonal='hist')
    for i in range(np.shape(axes)[0]):
        for j in range(np.shape(axes)[1]):
            if i < j:
                axes[i,j].set_visible(False)

    plt.show()

def plot_3d(data: Data):
    # x = np.linspace(0, 444*60, 500)
    #
    # fig = go.Figure()
    # fig.add_trace(go.Scatter3d(x=data.temp(x), y=data.light(x), z=data.res_time(x), mode='markers',
    #                              marker=dict(size=5, color=data.cat_conc(x), colorscale='Viridis', opacity=0.8, showscale=True)))
    #
    # # Add color bar
    # fig.update_layout(coloraxis=dict(colorbar=dict(title="Category Concentration", tickfont=dict(color='black', family='Arial', size=12))))
    #
    # # Update layout for axis titles
    # fig.update_layout(scene=dict(xaxis_title='Temperature', yaxis_title='Light Intensity', zaxis_title='Reaction Time'))
    #
    # # Update font style for axis titles
    # fig.update_layout(scene=dict(xaxis=dict(title_font=dict(color='black', family='Arial', size=14)),
    #                               yaxis=dict(title_font=dict(color='black', family='Arial', size=14)),
    #                               zaxis=dict(title_font=dict(color='black', family='Arial', size=14))))
    #
    # # fig.update_layout(scene=dict(bgcolor='white',
    # #                           xaxis=dict(showbackground=False, mirror=True, showline=True, color="black"),
    # #                           yaxis=dict(showbackground=False, gridcolor='white'),
    # #                           zaxis=dict(showbackground=False, gridcolor='white')))
    #
    # fig.layout.scene.xaxis.showline = True
    #
    # fig.show()

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation


    from mpl_toolkits.mplot3d import Axes3D
    x = np.linspace(0, 444*60, 300)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data.temp(x), data.light(x), data.res_time(x), c=data.D(x), cmap='viridis', s=5)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Light')
    ax.set_zlabel('Reaction Time')

    # Add color bar
    cbar = fig.colorbar(ax.scatter(data.temp(x), data.light(x), data.res_time(x), c=data.D(x), cmap='viridis'))
    cbar.set_label('Conversion')

    # Function to rotate the plot
    def update(num, ax, fig):
        ax.view_init(elev=10., azim=num)

    # Generate animation
    rotation_animation = FuncAnimation(fig, update, frames=np.arange(0, 360, 1), fargs=(ax, fig))

    # Save animation as gif
    # rotation_animation.save("D.gif", writer='imagemagick', fps=30, dpi=150)


    plt.show()

def main():
    path = pathlib.Path(r"C:\Users\nicep\Desktop\post_doc_2022\Data\polymerizations\DW2-14\data")
    data = Data(path)

    scatter_matrix_plot(data)
    # plot_3d(data)


if __name__ == "__main__":
    main()
