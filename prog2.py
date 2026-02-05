import matplotlib.pyplot as plt
import numpy as np

months = ["Jan","Feb","Mar","Apr","May","Jun", "Jul","Aug","Sep","Oct","Nov","Dec"]

face_cream = []
face_wash = []

print("Enter Face Cream sales for each month:")
for m in months:
    face_cream.append(float(input(f"Face Cream ({m}): ")))

print("\nEnter Face Wash sales for each month:")
for m in months:
    face_wash.append(float(input(f"Face Wash ({m}): ")))

x = np.arange(len(months))
width = 0.35

plt.bar(x - width/2, face_cream, width, label="Face Cream")
plt.bar(x + width/2, face_wash, width, label="Face Wash")

plt.xticks(x, months)
plt.title("Face Cream & Face Wash Sales per Month")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.legend()
plt.show()
