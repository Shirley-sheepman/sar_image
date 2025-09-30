# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 19:29:48 2025

@author: 28211
"""
import math
from nextpow import nextpow2
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm  # 进度条库



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


sig=np.zeros((Na,Nr));
for k in range(Npt):
    delay = 2/C*np.sqrt(Rpt[k]**2+(Ypt[k]-ta*V)**2)
    #print(delay)
    Dr=np.ones((Na,1))*tr.reshape(1,-1)-delay.reshape(-1,1)@np.ones((1,Nr))
    #sig=sig+math.exp(1j*pi*Kr*Dr)
    delay_col = delay.reshape(-1, 1) 
    phase_term = np.exp(1j * np.pi * Kr * Dr**2 - 1j * 2 * np.pi * fc * delay_col@np.ones((1,Nr)))
    azimuth_window = (np.abs(ta_col - Ypt[k]/V - Tc[k]) <= La[k] / (2 * V))
    range_window = (np.abs(Dr) <= Tr / 2)
    sig = sig + phase_term * azimuth_window * range_window
#
plt.figure()
plt.imshow(np.real(sig), extent=[col.min(), col.max(), row.min(), row.max()], 
           aspect='auto', origin='lower', cmap='hot')
plt.clim(0, 1)  # 对应caxis([0 1])
plt.colorbar()
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Real Part of Signal')
plt.show()


#脉冲压缩
sig_rd = np.fft.fft(sig, axis=1)

fr = np.linspace(-1/2, 1/2 - 1/Nr, Nr, endpoint=True)
fr = np.fft.fftshift(fr * Fr)
filter_r=np.ones((Na,1))@np.exp(1j*pi*fr**2/Kr).reshape(1,-1)

sig_rd=sig_rd*filter_r#匹配滤波
#上采样
nup=10
Nr_up=Nr*nup
nz = Nr_up-Nr;
dtr = 1/nup/Fr;
sig_rd_up=np.zeros((Na,Nr_up))
sig_rd_up[:, :Nr//2] = sig_rd[:, :Nr//2]
sig_rd_up[:, -Nr//2:] = sig_rd[:, Nr//2:]
sig_rdt = np.fft.ifft(sig_rd_up, axis=1)


plt.figure(figsize=(12, 6))
plt.imshow(np.abs(sig_rdt), aspect='auto', origin='lower', cmap='jet')
plt.colorbar(label='Amplitude')
plt.title(f'Distance Compressed Signal (Upsampled {nup}x)')
plt.xlabel(f'Range Samples ({Nr} → {Nr_up})')
plt.ylabel('Azimuth Samples')
plt.tight_layout()
plt.show()


#网格化
R=np.zeros((1,Nr))
for ii in range(Nr):
    R[0,ii]=R_left+(R_right-R_left)/(Nr-1)*(ii);
   
Y=np.zeros((1,Na))
for ii in range(Na):
    Y[0,ii]=Y_low+(Y_high-Y_low)/(Na-1)*(ii);
R=np.ones((Na,1))@R
Y=Y.reshape(-1, 1)@np.ones((1,Nr))


#成像
f_back=np.zeros((Na,Nr))

for ii in tqdm(range(Na), desc='BPA处理中'):
    R_ij=np.sqrt(R**2+(Y-V*ta[ii])**2)
    t_ij = 2 * R_ij / C
    
    # 3. 时间转换为采样点索引
    t_ij = np.round((t_ij - (2 * Rmin / C - Tr / 2)) / dtr).astype(int)
    
    # 4. 边界处理
    it_ij = (t_ij > 0) & (t_ij <= Nr_up)
    t_ij = t_ij * it_ij + Nr_up * (1 - it_ij.astype(int))
    
    # 5. 获取当前方位线的信号
    sig_rdta = sig_rdt[ii, :].copy()  # 使用copy避免修改原数据
    sig_rdta[Nr_up-1] = 0  # 边界置零
    
    # 6. 信号累加（核心步骤）
    # 使用高级索引获取对应距离门的信号值
    sig_values = sig_rdta[t_ij]
    
    # 相位补偿和累加
    phase_compensation = np.exp(1j * 4 * np.pi * R_ij / lamda)
    f_back =f_back+ sig_values * phase_compensation
print("BPA处理完成")



plt.figure(figsize=(10, 8))

# 显示图像（相当于MATLAB的imagesc）
plt.imshow(np.abs(f_back), 
           cmap='gray', 
           aspect='auto',
           extent=[0, f_back.shape[1], f_back.shape[0], 0])  # 设置坐标范围

plt.title('升采样率为10时成像结果', fontsize=14, fontproperties='SimHei')  # 支持中文
plt.xlabel('距离维', fontproperties='SimHei')
plt.ylabel('方位维', fontproperties='SimHei')
plt.colorbar(label='幅度')  # 添加颜色条

plt.tight_layout()
plt.show()




