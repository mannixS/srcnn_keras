import cv2 as cv
import glob

src_path = "D:\\python_project\\src\\label"
out_path = "D:\\python_project\\src\\label_chazhi"
imgs = glob.glob(src_path + '\\*.jpg')
for img in imgs:
    x = cv.imread(img, 0)
    x_expend = cv.resize(x, (1920, 1920), interpolation=4)
    img_name = img[img.rindex('\\') + 1:]
    cv.imwrite(out_path + '\\' + img_name, x_expend)
