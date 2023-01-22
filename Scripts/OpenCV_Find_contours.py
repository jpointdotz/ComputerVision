import cv2

def show_picture(img):
    cv2.imshow('shapes_and_colors', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_img():

    img = cv2.imread('shapes_and_colors.png')

    return img

def  contours(img):
    blurred = cv2.GaussianBlur(img, (5,5), 1)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    T, thresh = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)

    cnt, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cnt, hierarchy


def draw_contours(cnt, img, hierarchy):

    for contour in cnt:
        cv2.drawContours(img, [contour], -1, (0,255,0), 2)
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(img, (cx,cy), 3, (0,0,255),-1)
        area = cv2.contourArea(contour)
        cv2.putText(img, 'Area = {}'.format(area), (cx,cy-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)

    return img


# for contour in cnt:
#     (x,y),radius = cv2.minEnclosingCircle(contour)
#     center = (int(x),int(y))
#     radius = int(radius)
#     cv2.circle(img,center,radius,(0,255,0),2)

if __name__ == "__main__":
    image = read_img()
    show_picture(image)
    cnt, hierarchy = contours(image)
    image_contours = draw_contours(cnt, image, hierarchy)
    show_picture(image_contours)


