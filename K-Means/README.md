<center>
<img src="https://s2.ax1x.com/2019/01/23/kEMDne.png" width="60%" height="25%" />
</center>

### 算法思想：
```
选择K个点作为初始质心
repeat
    将每个点指派到最近的质心，形成K个簇
    均值法重新计算每个簇的质心
until 簇不发生变化或达到最大迭代次数
```
