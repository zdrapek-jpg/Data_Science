import numpy as np
def quickSort(data_array,left_idx,right_idx):
    if left_idx< right_idx:
        start = left_idx

        for i in range(left_idx+1,right_idx+1):
            if data_array[left_idx][0]>data_array[i][0]:
                start += 1
                temp = np.copy(data_array[i])
                data_array[i] = data_array[start]
                data_array[start] = temp

        temp = np.copy( data_array[left_idx])
        data_array[left_idx] = data_array[start]
        data_array[start]  = temp

        quickSort(data_array,left_idx,start-1)
        quickSort(data_array,start+1,right_idx)
    return data_array

