from math import sqrt
from numba import njit

def sq_distance(p1, p2):
    '''returns the squared distance between two points'''
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

@njit
def distance(p1, p2):
    '''returns the squared distance between two points'''
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
 

def pixels_to_cm(coords):
    x, y = coords[0], coords[1]
    x_new = (300/3478) * x
    y_new = (141/1630) * y
    new_pair = (x_new, y_new)
    return new_pair


def cm_radius_to_pixels(r):
    """Converts a radius in cm to a radius in pixels"""
    return r/((300/3478)**2+(141/1630)**2)**.5

def find_circle(max_pair, other_min, other_max):
    ''' src: https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/#:~:text=Equation%20of%20circle%20in%20general,and%20r%20is%20the%20radius.&text=Radius%20%3D%201-,The%20equation%20of%20the%20circle,2%20%2B%20y2%20%3D%201.'''
    max_pair_mid_point = (
        (max_pair[0][0] + max_pair[1][0]) / 2, (max_pair[0][1] + max_pair[1][1]) / 2)
    other_defining_point = max(
        [other_min, other_max], key=lambda p: distance(p, max_pair_mid_point))
    x1, y1 = max_pair[0][0], max_pair[0][1]
    x2, y2 = max_pair[1][0], max_pair[1][1]
    x3, y3 = other_defining_point[0], other_defining_point[1]
    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2)

    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2)

    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)

    f = (((sx13) * (x12) + (sy13) *
          (x12) + (sx21) * (x13) +
          (sy21) * (x13)) // (2 *
                              ((y31) * (x12) - (y21) * (x13))))

    g = (((sx13) * (y12) + (sy13) * (y12) +
          (sx21) * (y13) + (sy21) * (y13)) //
         (2 * ((x31) * (y12) - (x21) * (y13))))

    c = (-pow(x1, 2) - pow(y1, 2) -
         2 * g * x1 - 2 * f * y1)

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and
    # radius r as r^2 = h^2 + k^2 - c
    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c

    # r is the radius
    r = sqrt(sqr_of_r)
    return (h, k), r
