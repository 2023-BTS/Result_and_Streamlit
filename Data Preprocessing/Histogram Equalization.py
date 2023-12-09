import cv2

def BGR2RGB(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img

def HE_Color(image):
    img = cv2.cvtColor(BGR2RGB(image), cv2.COLOR_RGB2YCrCb) 
    img_planes = cv2.split(img) 
    img_planes_0 = cv2.equalizeHist(img_planes[0])
    merge_img = cv2.merge([img_planes_0, img_planes[1], img_planes[2]])
    he_img = cv2.cvtColor(merge_img, cv2.COLOR_YCrCb2RGB) 
    return he_img

def HE_Gray(image):
    img = cv2.cvtColor(BGR2RGB(image), cv2.COLOR_RGB2GRAY) 
    he_img = cv2.equalizeHist(img) 
    return he_img