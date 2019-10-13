import depth_manager as dm
from PIL import Image
img = Image.open("test.jpg")
label = "Apple"
v = dm.estimate_cal(img=img,label=label)
print(v)