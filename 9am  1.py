import matplotlib.pyplot as plt

#Data
subjects = ["Python", "Java", "SQL", "Power BI"]  
students = [40, 30, 20, 10]

# create Pie Chart
plt.pie(students, labels=subjects)

plt.title("students Ennrolled")
plt.show()