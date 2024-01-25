import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

def data_to_array(file):
    
    array = np.random.random ([24,32])*0 #Creates random array of size 24*32 and fills with zeros

    x = 0
    y = 0

    #For finding the average background if required
    sum1 = 0

    with open(file) as f:

        for line in f:

            #Seperating temperature value from unnecessary data in file
            
            if line[0] == '2': #Corresponding to format '2020-04-22 13:40:02,158 INFO: 17.764705882352942'
                head, sep, tail = line.partition('INFO: ')
            
            else: #Corresponding to format 'INFO:root:198'
                head, sep, tail = line.partition('INFO:root:')
                
            head, sep, tail = tail.partition('\n')

            temp = float(head) #Temperature value, taken from IR sensor file
            sum1 = sum1+temp

            array[y][x] = temp

            if x == 31:
                x = 0
                y = y+1

            else:
                x = x+1
                
    pixel_avg = sum1/(24*32) #If needed for calibration purposes etc.
            
    return(array, pixel_avg)


def subtract_array(array1, value):
    
    array = np.random.random ([24,32])*0
    
    if isinstance(value, np.ndarray): #If subtracting an array from an array
        for y in range(24):

            for x in range(32):

                array[y][x] = array1[y][x] - value[y][x]
                
    elif isinstance(value, float): #If subtracting a value from an array

        for y in range(24):

            for x in range(32):

                array[y][x] = array1[y][x] - value
    
    return(array)


temp_array = data_to_array("new.log")[0] #Can be edited for any IR data set

plt.figure(1)
ax = sns.heatmap(temp_array, linewidth=0, xticklabels=2, yticklabels=2, cmap="RdBu") #"RdBu_r" reverses colour gradient
plt.show()