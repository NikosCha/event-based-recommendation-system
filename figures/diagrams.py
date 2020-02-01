import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

def create_diagram(x_axis, y_axis1, y_axis2, x_axis_name, y_axis_name1, y_axis_name2, title_1, title_2, file_name, number):

    if number == 2:
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
    elif number == 1:
        plt.figure()
        plt.plot(x_axis, y_axis1)
        plt.ylabel(y_axis_name1)
        plt.xlabel(x_axis_name)
        plt.title(title_1)
        plt.grid(True)
        plt.savefig(file_name)

def create_diagram_multiple_cities(array1, array2, array3, array1_name, array2_name, array3_name, x_axis, x_axis_name, y_axis_name, title_1,file_name):
    plt.figure()
    
    plt.subplot(221)
    plt.plot(x_axis, array1, 'k--', label='array1_name')
    plt.plot(x_axis, array2, 'k:', label='array2_name')
    plt.plot(x_axis, array3, 'k', label='array3_name')
    legend = plt.legend(loc='upper left', shadow=False, fontsize='x-small')

    # Put a nicer background color on the legend.
    # legend.get_frame().set_facecolor('C0')

    plt.ylabel(y_axis_name)
    plt.xlabel(x_axis_name)

    plt.savefig(file_name)