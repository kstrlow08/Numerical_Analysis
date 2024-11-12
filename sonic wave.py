import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D 그래프를 위한 figure 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 축의 범위 설정
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])

# 축 레이블 추가
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 격자 추가
ax.grid(True)

# 보여주기
plt.show()
