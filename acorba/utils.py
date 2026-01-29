def Extract(lst,a): 
    return [item[0][a] for item in lst] 

def Diff(li1, li2):
    '''
    Return the different element comparing two lists
    '''
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))


def subtract_background(image, radius=50, light_bg=False):
    from skimage.morphology import white_tophat, black_tophat, disk
    str_el = disk(radius) #you can also use 'ball' here to get a slightly smoother result at the cost of increased computing time
    if light_bg:
        return black_tophat(image, str_el)
    else:
        return white_tophat(image, str_el)
