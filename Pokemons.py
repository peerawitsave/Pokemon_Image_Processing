import cv2 as cv
import numpy as np

print(cv.__version__)

image_gardevoir = cv.imread("Gardevoir.png")
image_gengar = cv.imread("Gengar.png")

gray_gardevoir = cv.cvtColor(image_gardevoir, cv.COLOR_BGR2GRAY)
gray_gengar = cv.cvtColor(image_gengar, cv.COLOR_BGR2GRAY)

def thresholding(image, threshold_val):
    _, binary_image = cv.threshold(image, threshold_val, 255, cv.THRESH_BINARY)
    return binary_image

def drawCircle(image, center_x, center_y, radius):
    cv.circle(image, (center_x, center_y), radius, (0, 0, 0), thickness=3)

binary_gardevoir = thresholding(gray_gardevoir, 128)
binary_gengar = thresholding(gray_gengar, 128)

height, width = gray_gardevoir.shape
combined_image = np.zeros((2 * height, 2 * width), dtype=np.uint8)

combined_image[0:height, 0:width] = gray_gardevoir          
combined_image[0:height, width:2*width] = binary_gardevoir  
combined_image[height:2*height, 0:width] = gray_gengar      
combined_image[height:2*height, width:2*width] = binary_gengar  

center_x = int(0.5 * width)
center_y = int(1.5 * height)
radius = min(width, height) // 4
drawCircle(combined_image, center_x, center_y, radius)

cv.imwrite("pokemon_collage.png", combined_image)
print("Image saved as pokemon_collage.png")

cv.imshow("Pokemon Collage", combined_image)
cv.waitKey(0)
cv.destroyAllWindows()
