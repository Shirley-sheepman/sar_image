# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:38:13 2025

@author: 28211
"""
import math
from nextpow import nextpow2
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm  # 进度条库

from scipy import interpolate

pi=math.pi

C=3e8
fc=5.3e9
lamda=C/fc
D=4
V=150
Kr=20e12 #调谐频率
Tr=2.5e-6 #一个chirp升频率时间
sq_ang=3.5/180*pi#斜视角

Br=Kr*Tr
Frfactor=1.2 #采样倍数
Fr=Br*Frfactor
Ba=0.886*2*V*math.cos(sq_ang)/D
Fa=1.2*Ba

R_near=2e4
R_far=R_near+1000


La_near=0.886*R_near*lamda/(math.cos(sq_ang)*math.cos(sq_ang))/D
La_far=0.886*R_far*lamda/(math.cos(sq_ang)*math.cos(sq_ang))/D
Tc_near = -R_near*math.tan(sq_ang)/V
Tc_far = -R_far*math.tan(sq_ang)/V
fdc = 2*V*math.sin(sq_ang)/lamda



Y_min = V*Tc_far
Y_max = Y_min+100;

Rmin = math.sqrt(R_near*R_near+(Tc_near*V+La_near/2)**2);
Rmax = math.sqrt(R_far*R_far+(Tc_far*V-La_far/2)**2);


Nr = (2*Rmax/C-2*Rmin/C+Tr)*Fr;
Nr = 2**nextpow2(Nr);
tr = np.linspace(-Tr/2+2*Rmin/C,Tr/2+2*Rmax/C,Nr)     # 快时间
Fr = (Nr-1)/(Tr/2+2*Rmax/C-(-Tr/2+2*Rmin/C));
Na = ((Tc_near+La_near/2/V)-(Tc_far-La_far/2/V))*Fa;
Na = 2**nextpow2(Na);
ta = np.linspace(Tc_far-La_far/2/V,Tc_near+La_near/2/V,Na)# 满时间
Fa = (Na-1)/(Tc_near+La_near/2/V-(Tc_far-La_far/2/V))

Rpt = [R_near, R_near+500 ,R_near+1000]
Rpt = np.array(Rpt)
Ypt = [0 ,0 ,0]
Ypt = np.array(Ypt)
La = 0.886*Rpt*lamda/math.cos(sq_ang)**2/D;
Tc = -Rpt*math.tan(sq_ang)/V;
Npt = len(Rpt);


Y_high = max(Ypt)+50;
Y_low = min(Ypt)-50;
# R_left = R_near-50;
# R_right = R_far+50;
R_left = Rmin;
R_right = Rmax;
row    = tr*C/2;                        #列
col    = ta*V; 

#matlab->python化处理
ta_col = ta.reshape(-1, 1)          # 形状 (Na, 1)


class FastBackprojection:
    """
    快速后向投影算法实现
    基于论文："Fast Backprojection Algorithm for Synthetic Aperture Radar"
    """
    
    def __init__(self, sig_rdt, ta, V, Rmin, Rmax, dtr, lamda, fc):
        """
        初始化参数
        """
        self.sig_rdt = sig_rdt  # 距离压缩后的信号
        self.ta = ta            # 方位时间序列
        self.V = V              # 平台速度
        self.Rmin = Rmin        # 最小距离
        self.Rmax = Rmax        # 最大距离
        self.dtr = dtr          # 距离采样间隔
        self.lamda = lamda      # 波长
        self.fc = fc            # 载波频率
        self.kc = 2 * np.pi / lamda  # 中心波数
        
        self.Na, self.Nr_up = sig_rdt.shape
        self.ds = V * (ta[1] - ta[0])  # 沿轨迹采样间隔
        
    def create_polar_grid(self, subap_center, subap_length, oversample_factor=1.2):
        """
        为子孔径创建极坐标网格
        根据论文公式(11)和(12)计算采样要求
        """
        # 计算最大频率（载频+带宽/2）
        nu_max = self.fc + Br / 2
        
        # 角度方向采样要求 (公式11)
        delta_alpha = C / (2 * nu_max * subap_length * self.ds)
        delta_alpha /= oversample_factor  # 过采样
        
        # 距离方向采样要求 (公式12)
        delta_r = C / (2 * Br)
        delta_r /= oversample_factor
        
        # 角度范围（余弦值，从-1到1）
        alpha_min, alpha_max = -1.0, 1.0
        n_alpha = int((alpha_max - alpha_min) / delta_alpha) + 1
        alphas = np.linspace(alpha_min, alpha_max, n_alpha)
        
        # 距离范围
        n_r = int((self.Rmax - self.Rmin) / delta_r) + 1
        rs = np.linspace(self.Rmin, self.Rmax, n_r)
        
        # 创建极坐标网格
        alpha_grid, r_grid = np.meshgrid(alphas, rs, indexing='ij')
        
        # 转换为笛卡尔坐标（相对于子孔径中心）
        x_rel = r_grid * alpha_grid
        y_rel = r_grid * np.sqrt(1 - alpha_grid**2)
        
        # 绝对坐标
        x_center = self.V * subap_center
        x_abs = x_center + x_rel
        y_abs = y_rel  # 假设飞行沿x轴，地面在y方向
        
        return {
            'alphas': alphas,
            'rs': rs,
            'alpha_grid': alpha_grid,
            'r_grid': r_grid,
            'x_abs': x_abs,
            'y_abs': y_abs,
            'x_rel': x_rel,
            'y_rel': y_rel
        }
    
    def backproject_to_polar_grid(self, subap_indices, polar_grid):
        """
        在极坐标网格上执行后向投影
        """
        subap_center = self.ta[len(subap_indices) // 2]
        polar_image = np.zeros(polar_grid['alpha_grid'].shape, dtype=complex)
        
        # 获取极坐标网格的绝对坐标
        x_abs = polar_grid['x_abs']
        y_abs = polar_grid['y_abs']
        
        for idx in tqdm(subap_indices, desc=f"处理子孔径脉冲", leave=False):
            t_current = self.ta[idx]
            x_radar = self.V * t_current
            
            # 计算所有极坐标点到当前雷达位置的距离
            R_ij = np.sqrt((x_abs - x_radar)**2 + y_abs**2)
            t_ij = 2 * R_ij / C
            
            # 转换为采样索引
            t_ij_idx = np.round((t_ij - (2 * self.Rmin / C - Tr / 2)) / self.dtr).astype(int)
            
            # 边界处理
            valid_mask = (t_ij_idx >= 0) & (t_ij_idx < self.Nr_up)
            t_ij_idx = np.where(valid_mask, t_ij_idx, 0)
            
            # 获取信号值
            sig_rdta = self.sig_rdt[idx, :].copy()
            sig_values = sig_rdta[t_ij_idx]
            sig_values[~valid_mask] = 0
            
            # 相位补偿和累加（包含载波相位）
            phase_comp = np.exp(1j * 4 * np.pi * R_ij / self.lamda)
            polar_image += sig_values * phase_comp
        
        return polar_image
    
    def remove_carrier_phase(self, polar_image, polar_grid):
        """
        去除载波相位（论文中的关键步骤）
        """
        # 去除极坐标距离对应的载波相位
        phase_remove = np.exp(-1j * 4 * np.pi * polar_grid['r_grid'] / self.lamda)
        return polar_image * phase_remove
    
    def upsample_polar_image(self, polar_image_no_phase, polar_grid, upsample_factor=4):
        """
        对去除相位的极坐标图像进行上采样
        """
        alphas = polar_grid['alphas']
        rs = polar_grid['rs']
        
        # 创建上采样后的网格
        n_alpha_up = len(alphas) * upsample_factor
        n_r_up = len(rs) * upsample_factor
        
        alphas_up = np.linspace(alphas[0], alphas[-1], n_alpha_up)
        rs_up = np.linspace(rs[0], rs[-1], n_r_up)
        
        # 双线性插值
        interp_func = interpolate.RegularGridInterpolator(
            (alphas, rs), polar_image_no_phase,
            method='linear', bounds_error=False, fill_value=0
        )
        
        alpha_grid_up, r_grid_up = np.meshgrid(alphas_up, rs_up, indexing='ij')
        points_up = np.column_stack((alpha_grid_up.ravel(), r_grid_up.ravel()))
        
        polar_image_up = interp_func(points_up).reshape(alpha_grid_up.shape)
        
        return {
            'image': polar_image_up,
            'alphas': alphas_up,
            'rs': rs_up,
            'alpha_grid': alpha_grid_up,
            'r_grid': r_grid_up
        }
    
    def interpolate_to_cartesian(self, polar_data, cartesian_grid, subap_center):
        """
        将上采样后的极坐标图像插值到笛卡尔网格
        """
        x_cart = cartesian_grid['X']
        y_cart = cartesian_grid['Y']
        
        # 计算笛卡尔网格点相对于子孔径中心的极坐标
        x_center = self.V * subap_center
        x_rel = x_cart - x_center
        r_cart = np.sqrt(x_rel**2 + y_cart**2)
        alpha_cart = x_rel / (r_cart + 1e-12)  # 避免除零
        
        # 创建插值函数
        interp_func = interpolate.RegularGridInterpolator(
            (polar_data['alphas'], polar_data['rs']), polar_data['image'],
            method='linear', bounds_error=False, fill_value=0
        )
        
        # 插值到笛卡尔网格
        points_cart = np.column_stack((alpha_cart.ravel(), r_cart.ravel()))
        image_no_phase = interp_func(points_cart).reshape(x_cart.shape)
        
        # 恢复载波相位
        phase_restore = np.exp(1j * 4 * np.pi * r_cart / self.lamda)
        subap_contribution = image_no_phase * phase_restore
        
        return subap_contribution
    
    def create_cartesian_grid(self, Y_low, Y_high, R_left, R_right, Na, Nr):
        """
        创建最终的笛卡尔网格
        """
        Y = np.linspace(Y_low, Y_high, Na)
        R = np.linspace(R_left, R_right, Nr)
        X, Y_grid = np.meshgrid(R, Y, indexing='xy')
        
        return {
            'X': X,
            'Y': Y_grid,
            'R': R,
            'Y_array': Y
        }
    
    def process(self, Y_low, Y_high, R_left, R_right, subap_size=None, upsample_factor=4):
        """
        执行快速后向投影处理
        """
        if subap_size is None:
            # 根据论文推荐选择最优子孔径大小
            subap_size = 8#max(1, int(np.sqrt(self.Na)))
        
        print(f"快速后向投影参数:")
        print(f"  子孔径大小: {subap_size} 脉冲")
        print(f"  上采样因子: {upsample_factor}")
        print(f"  总子孔径数: {self.Na // subap_size}")
        
        # 创建笛卡尔输出网格
        cartesian_grid = self.create_cartesian_grid(Y_low, Y_high, R_left, R_right, self.Na, self.Nr_up)
        final_image = np.zeros(cartesian_grid['X'].shape, dtype=complex)
        
        # 划分子孔径
        n_subaps = self.Na // subap_size
        subap_indices_list = [range(i * subap_size, (i + 1) * subap_size) 
                            for i in range(n_subaps)]
        
        # 处理每个子孔径
        for i, subap_indices in enumerate(tqdm(subap_indices_list, desc="处理子孔径")):
            if len(subap_indices) == 0:
                continue
                
            subap_center = self.ta[len(subap_indices) // 2 + subap_indices[0]]
            subap_length = len(subap_indices)
            
            # 1. 创建极坐标网格
            polar_grid = self.create_polar_grid(subap_center, subap_length)
            
            # 2. 在极坐标网格上执行后向投影
            polar_image = self.backproject_to_polar_grid(subap_indices, polar_grid)
            
            # 3. 去除载波相位
            polar_image_no_phase = self.remove_carrier_phase(polar_image, polar_grid)
            
            # 4. 上采样极坐标图像
            polar_data_up = self.upsample_polar_image(polar_image_no_phase, polar_grid, upsample_factor)
            
            # 5. 插值到笛卡尔网格并恢复相位
            subap_contribution = self.interpolate_to_cartesian(
                polar_data_up, cartesian_grid, subap_center
            )
            
            # 6. 累加到最终图像
            final_image += subap_contribution
        
        return final_image

def traditional_bp(sig_rdt, ta, V, Rmin, Rmax, dtr, lamda, Y_low, Y_high, R_left, R_right):
    """
    传统后向投影算法（用于对比）
    """
    Na, Nr_up = sig_rdt.shape
    
    # 建立成像网格
    R = np.linspace(R_left, R_right, Nr_up)
    Y = np.linspace(Y_low, Y_high, Na)
    R_grid, Y_grid = np.meshgrid(R, Y)
    
    f_back = np.zeros((Na, Nr_up), dtype=complex)
    
    print("传统BP算法处理中...")
    for ii in tqdm(range(Na)):
        R_ij = np.sqrt(R_grid**2 + (Y_grid - V * ta[ii])**2)
        t_ij = 2 * R_ij / C
        t_ij_idx = np.round((t_ij - (2 * Rmin / C - Tr / 2)) / dtr).astype(int)
        
        valid_mask = (t_ij_idx >= 0) & (t_ij_idx < Nr_up)
        t_ij_idx = np.where(valid_mask, t_ij_idx, 0)
        
        sig_rdta = sig_rdt[ii, :].copy()
        sig_values = sig_rdta[t_ij_idx]
        sig_values[~valid_mask] = 0
        
        phase_comp = np.exp(1j * 4 * np.pi * R_ij / lamda)
        f_back += sig_values * phase_comp
    
    return f_back



def main():
    # 生成回波信号
    print("生成回波信号...")
    sig = np.zeros((Na, Nr), dtype=complex)
    for k in range(Npt):
        delay = 2 / C * np.sqrt(Rpt[k]**2 + (Ypt[k] - ta * V)**2)
        Dr = np.ones((Na, 1)) * tr.reshape(1, -1) - delay.reshape(-1, 1) @ np.ones((1, Nr))
        delay_col = delay.reshape(-1, 1)
        phase_term = np.exp(1j * np.pi * Kr * Dr**2 - 1j * 2 * np.pi * fc * delay_col @ np.ones((1, Nr)))
        azimuth_window = (np.abs(ta_col - Ypt[k] / V - Tc[k]) <= La[k] / (2 * V))
        range_window = (np.abs(Dr) <= Tr / 2)
        sig = sig + phase_term * azimuth_window * range_window

    # 脉冲压缩
    print("进行脉冲压缩...")
    sig_rd = np.fft.fft(sig, axis=1)
    fr = np.linspace(-1 / 2, 1 / 2 - 1 / Nr, Nr, endpoint=True)
    fr = np.fft.fftshift(fr * Fr)
    filter_r = np.ones((Na, 1)) @ np.exp(1j * pi * fr**2 / Kr).reshape(1, -1)
    sig_rd = sig_rd * filter_r

    # 上采样
    nup = 10
    Nr_up = Nr * nup
    dtr = 1 / nup / Fr
    sig_rd_up = np.zeros((Na, Nr_up), dtype=complex)
    sig_rd_up[:, :Nr // 2] = sig_rd[:, :Nr // 2]
    sig_rd_up[:, -Nr // 2:] = sig_rd[:, Nr // 2:]
    sig_rdt = np.fft.ifft(sig_rd_up, axis=1)

    # 性能对比
    import time
    
    print("\n=== 传统后向投影算法 ===")
    start_time = time.time()
    image_traditional = traditional_bp(sig_rdt, ta, V, Rmin, Rmax, dtr, lamda, Y_low, Y_high, R_left, R_right)
    trad_time = time.time() - start_time
    
    
    # 快速BP
    print("\n=== 快速后向投影算法 ===")
    fbp = FastBackprojection(sig_rdt, ta, V, Rmin, Rmax, dtr, lamda, fc)
    
    start_time = time.time()
    image_fast = fbp.process(Y_low, Y_high, R_left, R_right, upsample_factor=4)
    fast_time = time.time() - start_time
    print(f"快速BP耗时: {fast_time:.2f}秒")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 传统BP结果
    im0 = axes[0].imshow(np.abs(image_traditional), cmap='gray', aspect='auto',
                        extent=[0, image_traditional.shape[1], image_traditional.shape[0], 0])
    axes[0].set_title(f'传统BP算法\n耗时: {trad_time:.2f}秒', fontproperties='SimHei')
    axes[0].set_xlabel('距离维', fontproperties='SimHei')
    axes[0].set_ylabel('方位维', fontproperties='SimHei')
    plt.colorbar(im0, ax=axes[0])
    
    # 快速BP结果
    im1 = axes[1].imshow(np.abs(image_fast), cmap='gray', aspect='auto',
                        extent=[0, image_fast.shape[1], image_fast.shape[0], 0])
    axes[1].set_title(f'快速BP算法\n耗时: {fast_time:.2f}秒', fontproperties='SimHei')
    axes[1].set_xlabel('距离维', fontproperties='SimHei')
    axes[1].set_ylabel('方位维', fontproperties='SimHei')
    plt.colorbar(im1, ax=axes[1])
    
    # 残差图像
    residual = np.abs(image_traditional - image_fast)
    im2 = axes[2].imshow(residual, cmap='hot', aspect='auto',
                        extent=[0, residual.shape[1], residual.shape[0], 0])
    axes[2].set_title(f'残差图像\n最大残差: {np.max(residual):.2e}', fontproperties='SimHei')
    axes[2].set_xlabel('距离维', fontproperties='SimHei')
    axes[2].set_ylabel('方位维', fontproperties='SimHei')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


































