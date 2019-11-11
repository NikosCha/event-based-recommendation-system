import matplotlib.pyplot as plt

def create_diagram(x_axis, y_axis1, y_axis2, x_axis_name, y_axis_name1, y_axis_name2, title_1, title_2, file_name):
    plt.figure()
    
    plt.subplot(221)
    plt.plot(x_axis, y_axis1)
    plt.ylabel(y_axis_name1)
    plt.xlabel(x_axis_name)
    plt.title(title_1)
    plt.grid(True)

    plt.subplot(222)
    plt.plot(x_axis, y_axis2)
    plt.ylabel(y_axis_name2)
    plt.xlabel(x_axis_name)
    plt.title(title_2)
    plt.grid(True)
    # Format the minor tick labels of the y-axis into empty strings with

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.15, right=0.95, hspace=0.25,
                        wspace=0.35)

    plt.savefig(file_name)