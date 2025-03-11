# -Virtual-Power-Plant-
1)在时间维度上，采用动态规划求解，在空间维度上，运用改进的Shapley值法量化各参与方的边际贡献度，引入风险偏好因子修正传统分配方案；3)在动态博弈层面，构建纳什议价模型，设计基于交替方向乘子法(ADMM)的分布式求解算法。
caiyang.py文件是图2.2蒙特卡洛(Monte Carlo)与拉丁超立方抽样（LHS）风速光照强度对比图与图2.3 蒙特卡洛(Monte Carlo)与拉丁超立方抽样（LHS）收敛性分析图的运行代码。数据来源于文件POWER_Parameter_Manager.csv文件,该数据来源于网站：https://power.larc.nasa.gov/data-access-viewer/
文件TNCO.py是三层嵌套优化的完整实现代码。
