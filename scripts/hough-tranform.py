import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_average_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    negative_slopes = []
    positive_slopes = []
    
    negetive_intercepts = []
    positive_intercepts = []
    
    left_line_x = []
    left_line_y = []
    
    right_line_x = []
    right_line_y = []
    
    y_max = img.shape[0]
    y_min = img.shape[0]
    
    # The idea of this algorithm is 
    # 1. separate line segments into left and right lines
    # 2. Then for each line
    #    2.1 calculate average slope and intercept
    # 3. considering all line segments calculate y_min value
    # 4. Set height of the image into the y_max value
    # 5. For both left and right lines:
    #   5.1 calculate x_min and x_max values using intercept, slope, y_min and y_max
    #   5.2 Draw a line using (x_min, y_min) and (x_max, y_max)
    for line in lines:
        for x1,y1,x2,y2 in line:
            current_slope = (y2-y1)/(x2-x1)
            
            if current_slope < 0.0 and current_slope > -math.inf:
                negative_slopes.append(current_slope) # left line
                left_line_x.append(x1)
                left_line_x.append(x2)
                left_line_y.append(y1)
                left_line_y.append(y2)
                negetive_intercepts.append(y1 -current_slope*x1)
            
            if current_slope > 0.0 and current_slope < math.inf:
                positive_slopes.append(current_slope) # right line
                right_line_x.append(x1)
                right_line_x.append(x2)
                right_line_y.append(y1)
                right_line_y.append(y2)
                positive_intercepts.append(y1 - current_slope*x1)
                
            y_min = min(y_min, y1, y2)
                    
    y_min += 15 # add small threshold
    
    if len(positive_slopes) > 0 and len(right_line_x) > 0 and len(right_line_y) > 0:
        ave_positive_slope = sum(positive_slopes) / len(positive_slopes)
        ave_right_line_x = sum(right_line_x) / len(right_line_x)
        ave_right_line_y = sum(right_line_y ) / len(right_line_y)
        intercept = sum(positive_intercepts) / len(positive_intercepts)    
        x_min=int((y_min-intercept)/ave_positive_slope) 
        x_max = int((y_max - intercept)/ ave_positive_slope)
        cv2.line(img, (x_min, y_min), (x_max, y_max), [255, 0, 0], 12)

    if len(negative_slopes) > 0 and len(left_line_x) > 0 and len(left_line_y) > 0:
        ave_negative_slope = sum(negative_slopes) / len(negative_slopes)
        ave_left_line_x = sum(left_line_x) / len(left_line_x)
        ave_left_line_y = sum(left_line_y ) / len(left_line_y)
        intercept = sum(negetive_intercepts) / len(negetive_intercepts)
        x_min = int((y_min-intercept)/ave_negative_slope) 
        x_max = int((y_max - intercept)/ ave_negative_slope)
        # cv2.line(img, (x_min, y_min), (x_max, y_max), [122, 1, 255], 12)
        cv2.line(img, (x_min, y_min), (x_max, y_max), [255, 0, 0], 10)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def hough_ave_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_average_lines(line_img, lines, color=[0,0,255],thickness=8)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

# Read in and grayscale the image
image = mpimg.imread('solidWhiteCurve.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
cv2.imwrite('../test_images_output/gray.png',gray)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 3
blur_gray = gaussian_blur(image,kernel_size)
cv2.imwrite('../test_images_output/gaussian_blur.png',cv2.cvtColor(blur_gray, cv2.COLOR_RGB2BGR))

# Define our parameters for Canny and apply
low_threshold = 75
high_threshold = 150
edges = canny(blur_gray, low_threshold, high_threshold)
cv2.imwrite('../test_images_output/edges.png',edges)

# plt.imshow(edges)
# plt.show()

# This time we are defining a four sided polygon to mask
# region_select = np.copy(image)
imshape = image.shape
height=imshape[0]
width=imshape[1]
bottom_off = 75
top_off = 40
# vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
vertices = np.array([[(bottom_off,height),(width/2-top_off, height/2+top_off), 
                        (width/2+top_off, height/2+top_off), (width-bottom_off,height)]], dtype=np.int32)

region_selected = region_of_interest(edges, vertices)
cv2.imwrite('../test_images_output/region_selected.png',region_selected)

# plt.imshow(region_selected)
# plt.show()
# cv2.fillPoly(mask, vertices, ignore_mask_color)
# masked_edges = cv2.bitwise_and(edges, mask)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 100      # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40 #minimum number of pixels making up a line
max_line_gap = 25   # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
hough = hough_lines(region_selected, rho, theta, threshold, min_line_length, max_line_gap)
hough_ave_alone = hough_ave_lines(region_selected, rho, theta, threshold, min_line_length, max_line_gap)

lines_ave = cv2.addWeighted(hough_ave_alone, 0.6, hough, 1, 0)

# lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
#                             min_line_length, max_line_gap)


cv2.imwrite('../test_images_output/hough.png',cv2.cvtColor(hough, cv2.COLOR_RGB2BGR))
cv2.imwrite('../test_images_output/hough_ave.png',cv2.cvtColor(lines_ave, cv2.COLOR_RGB2BGR))

# plt.imshow(hough)
# plt.show()
# Create a "color" binary image to combine with line image
# color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
lines_edges = cv2.addWeighted(hough, 0.9, image, 1, 0)
lines_ave = cv2.addWeighted(hough_ave_alone, 0.9, image, 1, 0)

plt.imshow(lines_edges)
plt.show()
cv2.imwrite('../test_images_output/result.png',cv2.cvtColor(lines_edges, cv2.COLOR_RGB2BGR))
cv2.imwrite('../test_images_output/result_ave.png',cv2.cvtColor(lines_ave, cv2.COLOR_RGB2BGR))

