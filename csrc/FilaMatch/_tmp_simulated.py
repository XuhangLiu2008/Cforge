import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import collections

class OptimizedPearliteSimulator:
    def __init__(self, size=500, num_seeds=25):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.vector_field = np.zeros((size, size, 2))
        
        # 核心优化：使用集合存储生长沿像素 (r, c)
        self.active_surface = collections.deque()
        
        # 状态与颜色 (5种相)
        self.AUSTENITE = 0
        self.colors = ['#080808', '#C0C0C0', '#4B0082', '#FFD700', '#00FA9A', '#1E90FF']
        
        # 预计算周期场 (方案1: 周期随空间变化)
        x = np.linspace(0, 1, size)
        X, _ = np.meshgrid(x, x)
        self.period_map = 10.0 - 4.0 * X 
        
        self._initialize_seeds(num_seeds)

    def _initialize_seeds(self, num_seeds):
        for _ in range(num_seeds):
            r, c = np.random.randint(10, self.size-10, 2)
            angle = np.random.uniform(0, 2 * np.pi)
            v = np.array([np.cos(angle), np.sin(angle)])
            
            self.grid[r, c] = 1 
            self.vector_field[r, c] = v
            self.active_surface.append((r, c))

    def get_phase_ratios(self, r, c):
        """浓度变化逻辑：决定各相宽度占比"""
        x_norm = c / self.size
        r1 = max(0.1, 0.8 - 0.6 * x_norm)   # 基础相
        r3 = 0.3 * x_norm                  # 随浓度析出的合金相1
        r4 = 0.2 * (x_norm**2)             # 随浓度析出的合金相2
        r2 = max(0, 1.0 - (r1 + r3 + r4))  # 剩余相
        return [r1, r2, r3, r4]

    def update(self, frame):
        if not self.active_surface:
            return [self.im]

        # 每帧生长的点数，可以根据需要调整生长速度
        growth_rate = len(self.active_surface) 
        new_active = collections.deque()
        
        # 随机打乱当前生长沿
        np.random.shuffle(self.active_surface)
        
        for _ in range(growth_rate):
            if not self.active_surface: break
            r, c = self.active_surface.popleft()
            
            has_austenite_neighbor = False
            parent_v = self.vector_field[r, c]
            
            # 检查邻域
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.grid[nr, nc] == self.AUSTENITE:
                        has_austenite_neighbor = True
                        
                        # 1. 局部属性继承与扰动
                        v = parent_v + np.random.normal(0, 0.0002, 2)
                        v /= np.linalg.norm(v)
                        self.vector_field[nr, nc] = v
                        
                        # 2. 获取该点局部参数
                        ratios = self.get_phase_ratios(nr, nc)
                        local_period = self.period_map[nr, nc]
                        
                        # 3. 共享规则判定
                        proj = nr * v[0] + nc * v[1]
                        rel_pos = proj % local_period
                        
                        cum_r = 0
                        for idx, ratio in enumerate(ratios):
                            cum_r += ratio * local_period
                            if rel_pos < cum_r:
                                self.grid[nr, nc] = idx + 1
                                break
                        
                        # 4. 新结晶点加入生长沿
                        new_active.append((nr, nc))
            
            # 如果该点还有未结晶邻居，保留在生长沿中
            if has_austenite_neighbor:
                new_active.append((r, c))
        
        self.active_surface = new_active
        self.im.set_array(self.grid)
        return [self.im]

    def run(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = ListedColormap(self.colors)
        self.im = ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=5, interpolation='nearest')
        ax.set_title("优化版: 生长沿维护 + 周期梯度调制", color='white')
        fig.patch.set_facecolor('#121212')
        ax.axis('off')
        
        # 帧数设多一些，因为现在只算边缘，速度极快
        ani = FuncAnimation(fig, self.update, frames=300, interval=20, blit=True)
        plt.show()

sim = OptimizedPearliteSimulator()
sim.run()