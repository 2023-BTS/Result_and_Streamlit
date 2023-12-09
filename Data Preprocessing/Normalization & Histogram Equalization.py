import cv2

def BGR2RGB(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img

def Normalization_Color(image):
    normalize_img = cv2.normalize(BGR2RGB(image), None, 0, 255, cv2.NORM_MINMAX)
    return normalize_img

def Normalization_Gray(image):
    img = cv2.cvtColor(BGR2RGB(image), cv2.COLOR_RGB2GRAY)
    normalize_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return normalize_img

def NHE_Color(image):
    img = cv2.cvtColor(Normalization_Color(image), cv2.COLOR_RGB2YCrCb)
    img_planes = cv2.split(img) 
    img_planes_0 = cv2.equalizeHist(img_planes[0])
    merge_img = cv2.merge([img_planes_0, img_planes[1], img_planes[2]])
    nhe_img = cv2.cvtColor(merge_img, cv2.COLOR_YCrCb2RGB)
    return nhe_img

def NHE_Gray(image):
    nhe_img = cv2.equalizeHist(Normalization_Gray(image))
    return nhe_img