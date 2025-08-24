import cv2
import matplotlib.pyplot as plt

# โหลดภาพ (BGR)
img = cv2.imread(r"C:\Mango-Disease-70_15_15\test\Anthracnose\image_Anthracnose_25.jpg")

# แปลงเป็น LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# สร้าง CLAHE object
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)

# รวมกลับ
limg = cv2.merge((cl,a,b))
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# แสดงผล
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Before CLAHE")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,2,2)
plt.title("After CLAHE")
plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
