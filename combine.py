import  copy
import cv2 as cv


def combine_two_color_images_with_anchor(image1, image2, anchor_y, anchor_x):
    foreground, background = image1.copy(), image2.copy()
    # Check if the foreground is inbound with the new coordinates and raise an error if out of bounds
    background_height = background.shape[0]
    background_width = background.shape[1]
    print(background_height )
    print(background_width)
    foreground_height = foreground.shape[0]
    foreground_width = foreground.shape[1]
    if foreground_height+anchor_y > background_height or foreground_width+anchor_x > background_width:
        raise ValueError("The foreground image exceeds the background boundaries at this location")

    alpha =0.5

    # do composite at specified location
    start_y = anchor_y
    start_x = anchor_x
    end_y = anchor_y+foreground_height
    end_x = anchor_x+foreground_width
    blended_portion = cv.addWeighted(foreground,
                alpha,
                background[start_y:end_y, start_x:end_x,:],
                1 - alpha,
                0,
                background)
    background[start_y:end_y, start_x:end_x,:] = blended_portion
    cv.imshow('composited image', background)
    cv.imwrite('op.jpg',background)

    cv.waitKey(10000)


image2=cv.imread('try1.jpg')
image1=cv.imread('d1.jpg')

combine_two_color_images_with_anchor(image1,image2,0,0) 