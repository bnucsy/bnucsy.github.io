---
title: A Learned Compact and Editable Light Field Representation
categories: [paper, cv]
mathjax: true
date: 2022-08-13 09:31:37
---

## [A Learned Compact and Editable Light Field Representation](https://arxiv.org/abs/2103.11314)【光场压缩、编辑传播】【arxiv】

<center>P.S. 本文为自己重述，不完全忠实于原paper，难免出现错误</center>

&emsp;&emsp;该论文完成了光场数据的压缩表示，使用自编码器的方法将光场数据压缩为一个深度图和一个中央视觉图，并且支持编辑之后的中央视觉图和原深度图进行光场重建。

<img src='https://img-blog.csdnimg.cn/f7655d1624894b87a117e25801a7b14d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAYm51Y3N5,size_20,color_FFFFFF,t_70,g_se,x_16' align='center' style="zoom: 80%;" ></img>

<!-- more -->

### 简介

 &emsp; &emsp; 所谓光场，即记录光线位置和方向的数据结构，这种数据表示方式经历了很大的变化历程，相关的介绍中<a href='https://www.leiphone.com/category/arvr/N14i2K6UmZzK5TcE.html'>这个介绍</a>最为清晰和全面。
&emsp; &emsp;本文对于这种数据结构进行了自编码器的学习，综合考虑解码后与原图的结构差异，编码的中间层中央视觉图部分的RGB相似度，编辑后的中央视觉图的重建程度，完成了一种支持编辑的光场重建。这项工作的压缩部分解决了光场数据结构较大的问题，可编辑重建部分解决了二维图像处理到三维图像处理的迁移问题，总体来说意义重大。目前本文发布于<a href='https://arxiv.org/abs/2103.11314'> arxiv</a>，也许在不久之后会登上SIGGRAPH。

### 相关工作


 - 光场重建相关
&emsp; &emsp; 以往的光场重建工作主要集中于：①<a href='https://dl.acm.org/doi/10.1145/2682631#d32752538e1'>从稀疏样本合成密集光场。</a>②<a href='https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6574844'>通过光场估计视差。</a>③<a href='https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7406366'>对光场合成的深度学习方法——该方法可以估计扭曲的视差，并更细致地描绘颜色。</a>④<a href='https://arxiv.org/pdf/1708.03292.pdf'>通过单张照片合成光场。</a>⑤<a href='https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953925'>多平面图像表示光场。</a>
&emsp; &emsp;这些工作尽管能够解决部分问题，但仍有局限，如前几个工作会出现某些伪影、重影。最后一类工作引出一个新的问题——如何通过编辑这种多平面图象进行光场编辑。
&emsp; &emsp;我认为本文的工作在某种程度上重复了④的工作内容，类似于AE和GAN之间的联系。在完成④的基础上，本文解决了⑤引出的可编辑性问题——通过分离特征和视图的方法。

 - 光场编辑相关
&emsp; &emsp;以往的这些工作集中于：①目标重定位。②目标变形。③目标补充。④<a href='http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=36554C114E54C1449058A76D6684D6AB?doi=10.1.1.644.3384&rep=rep1&type=pdf'>将稀疏编辑传播至全光场。</a>⑤<a href='https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7414488'>将中心视图分解为不同深度的图层，通过图层进行光场编辑。</a>
&emsp; &emsp;上述工作中④，⑤两个工作相比而言较为接近本文工作，因此没有给出①②③的原文链接，尤其是④，④最关键的难度在于进行光场编辑的一致性传播，与本文想法接近。有较强的借鉴意义。

 &emsp; 
### 本文的工作

 &emsp; &emsp;总体来说，本文能够对一个给定的4D光场，将其表示为一个元通道`在没有限制元通道的视觉形态的情况下，为什么它长得这么像一个灰度图？`和一个RGB图（视觉通道），这实现了4D光场数据的压缩，另一方面，针对视觉通道的编辑可以通过和元通道的结合重建光场，并一致性地传播到新的光场之中。
本文设计的网络结构：<img src='https://img-blog.csdnimg.cn/a35c15d6728748e487dc22097d88cf4c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAYm51Y3N5,size_20,color_FFFFFF,t_70,g_se,x_16' width=50% align=center>

1. #### 问题描述
   &emsp; &emsp;输入的4D光场视图即为$M\times N$张RGB图像，我们描述为$L = \{I_i\}_{i=1}^{M\times N}$，对于这个光场$L$,$L(u,v)$也就表示了某一个像素，其中$u$表示图片，$v$表示像素，也就是说$I_i$即为$L(u_i)$。

   &emsp; &emsp;在整个自编码器的设计过程之中，我们记编码器为$E$，解码器为$D$。

   &emsp; &emsp;一方面，对于输入的光场$L$，我们通过编码器生成一个元通道$Z$，即$Z = E(L)$，有$sizeof(Z) = sizeof(I_i)$。另一方面，对于传入的光场数据，我们获取中央视图$I_c$ `Ic具体是如何得到的呢？是L(x,0)么？`。对$I_c$进行编辑之后我们获得$\widetilde{I}_c$，接着我们把两者结合，通过解码网络获得编辑后光场$\widetilde L$，即 $\widetilde L = D(Z, \widetilde I _c)$。

2. #### 表示方式的选取
   &emsp; &emsp;本节实际上主要阐述了为什么要选取元通道和视觉通道。

   &emsp; &emsp;中心视图 $I_c$ 捕获了光场的参考视觉内容，因此自编码器只需要学习表示元通道即可。元通道作为视觉通道互补的部分，意味着包括深度、视差、纹理、遮挡等等信息被隐式地编码到元通道之中。事实上本文做了[消融实验](https://blog.csdn.net/flyfish1986/article/details/104812229#:~:text=%E6%B6%88%E8%9E%8D%E5%AE%9E%E9%AA%8C%20%EF%BC%88%20ablation%20experiment%EF%BC%89,%E5%9C%A8%E6%9C%BA%E5%99%A8%20%E5%AD%A6%E4%B9%A0%20%EF%BC%8C%E7%89%B9%E5%88%AB%E6%98%AF%E5%A4%8D%E6%9D%82%E7%9A%84%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E8%83%8C%E6%99%AF%E4%B8%8B%EF%BC%8C%E5%B7%B2%E7%BB%8F%E9%87%87%E7%94%A8%E2%80%9C%20%E6%B6%88%E8%9E%8D%E7%A0%94%E7%A9%B6%20%E2%80%9D%E6%9D%A5%E6%8F%8F%E8%BF%B0%E5%8E%BB%E9%99%A4%E7%BD%91%E7%BB%9C%E7%9A%84%E6%9F%90%E4%BA%9B%E9%83%A8%E5%88%86%E7%9A%84%E8%BF%87%E7%A8%8B%EF%BC%8C%E4%BB%A5%E4%BE%BF%E6%9B%B4%E5%A5%BD%E5%9C%B0%E7%90%86%E8%A7%A3%E7%BD%91%E7%BB%9C%E7%9A%84%E8%A1%8C%E4%B8%BA%E3%80%82)，使用深度图代替了元通道，取得了较差的结果。

   中央视觉图、元通道、深度图：<img src='https://img-blog.csdnimg.cn/ae4a919d4c164d27b71ae77dd83e9f50.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAYm51Y3N5,size_20,color_FFFFFF,t_70,g_se,x_16' width=50% align='center'></img>

   &emsp; &emsp;关于元通道的个数和是否必须存在的问题，本文解释道，当图片数目（角度分辨率）和图片大小（空间分辨率）增加的时候，单个元通道可能无法编码全部信息，这时候我们可以增加元通道的数目。至于和视觉图分离的原因，主要是考虑到光场的重建过程中对于编辑的视觉通道，我们需要原光场的结构，如果将元通道编码到视觉通道之中形成 $I^z$，那么编辑的过程就会破坏结构，光场不再能够重建。

   &emsp; &emsp;事实上，如果我们不考虑光场的重建问题，我们是可以将视觉通道和元通道合二为一的，此时我们只考虑数据压缩和光场生成的任务，该方法仍旧是一个很好的想法。这也将作为我的毕业设计的主要思想。

3. #### 编辑敏感的光场重建（解码器网络实现）

   &emsp; &emsp;本文将解码器的解释信息的方式分为两类：视图间视差图，以及其他信息（遮挡图和[非朗博效应](https://www.cnblogs.com/qingsunny/archive/2013/03/07/2947572.html#:~:text=%E5%AF%B9%E9%9D%9E%E6%9C%97%E4%BC%AF%E4%BD%93%E8%80%8C%E8%A8%80%EF%BC%8C%E5%AE%83%E5%AF%B9%E5%A4%AA%E9%98%B3%E7%9F%AD%E6%B3%A2%E8%BE%90%E5%B0%84%E7%9A%84%E5%8F%8D%E5%B0%84%E3%80%81%E6%95%A3%E5%B0%84%E8%83%BD%E5%8A%9B%E4%B8%8D%E4%BB%85%E9%9A%8F%E6%B3%A2%E9%95%BF%E8%80%8C%E5%8F%98%EF%BC%8C%E5%90%8C%E6%97%B6%E4%BA%A6%E9%9A%8F%E7%A9%BA%E9%97%B4%E6%96%B9%E5%90%91%E8%80%8C%E5%8F%98%E3%80%82%20%E6%89%80%E8%B0%93%E5%9C%B0%E7%89%A9%E7%9A%84%E6%B3%A2%E8%B0%B1%E7%89%B9%E5%BE%81%E6%98%AF%E6%8C%87%E8%AF%A5%E5%9C%B0%E7%89%A9%E5%AF%B9%E5%A4%AA%E9%98%B3%E8%BE%90%E5%B0%84%E7%9A%84%E5%8F%8D%E5%B0%84%E3%80%81%E6%95%A3%E5%B0%84%E8%83%BD%E5%8A%9B%E9%9A%8F%E6%B3%A2%E9%95%BF%E8%80%8C%E5%8F%98%E7%9A%84%E8%A7%84%E5%BE%8B%E3%80%82%20%E5%9C%B0%E7%89%A9%E6%B3%A2%E8%B0%B1%E7%89%B9%E5%BE%81%E4%B8%8E%E5%9C%B0%E7%89%A9%E7%9A%84%E7%BB%84%E6%88%90%E6%88%90%E4%BB%BD%EF%BC%8C%E7%89%A9%E4%BD%93%E5%86%85%E9%83%A8%E7%9A%84%E7%BB%93%E6%9E%84%E5%85%B3%E7%B3%BB%E5%AF%86%E5%88%87%EF%BC%8C%E9%80%9A%E4%BF%97%E8%AE%B2%E5%9C%B0%E7%89%A9%E6%B3%A2%E8%B0%B1%E7%89%B9%E5%BE%81%E4%B9%9F%E5%B0%B1%E6%98%AF%E5%9C%B0%E7%89%A9%E7%9A%84%E9%A2%9C%E8%89%B2%E7%89%B9%E5%BE%81%E3%80%82%20%E8%80%8C%E5%9C%B0%E7%89%A9%E7%9A%84%E6%96%B9%E5%90%91%E7%89%B9%E5%BE%81%E6%98%AF%E7%94%A8%E6%9D%A5%E6%8F%8F%E8%BF%B0%E5%9C%B0%E7%89%A9%E5%AF%B9%E5%A4%AA%E9%98%B3%E8%BE%90%E5%B0%84%E5%8F%8D%E5%B0%84%E3%80%81%E6%95%A3%E5%B0%84%E8%83%BD%E5%8A%9B%E5%9C%A8%E6%96%B9%E5%90%91%E7%A9%BA%E9%97%B4%E5%8F%98%E5%8C%96%E7%9A%84%EF%BC%8C%E8%BF%99%E7%A7%8D%E7%A9%BA%E9%97%B4%E5%8F%98%E5%8C%96%E7%89%B9%E5%BE%81%E4%B8%BB%E8%A6%81%E5%86%B3%E5%AE%9A%E4%BA%8E%E4%B8%A4%E7%A7%8D%E5%9B%A0%E7%B4%A0%EF%BC%8C%E5%85%B6%E4%B8%80%E6%98%AF%E7%89%A9%E4%BD%93%E7%9A%84%E8%A1%A8%E9%9D%A2%E7%B2%97%E7%B3%99%E5%BA%A6%EF%BC%8C%E5%AE%83%E4%B8%8D%E4%BB%85%E5%8F%96%E5%86%B3%E4%BA%8E%E8%A1%A8%E9%9D%A2%E5%B9%B3%E5%9D%87%E7%B2%97%E7%B3%99%E9%AB%98%E5%BA%A6%E5%80%BC%E4%B8%8E%E7%94%B5%E7%A3%81%E6%B3%A2%E6%B3%A2%E9%95%BF%E4%B9%8B%E9%97%B4%E7%9A%84%E6%AF%94%E4%BE%8B%E5%85%B3%E7%B3%BB%EF%BC%8C%E8%80%8C%E4%B8%94%E8%BF%98%E4%B8%8E%E8%A7%86%E8%A7%92%E5%85%B3%E7%B3%BB%E5%AF%86%E5%88%87%E3%80%82%20%E8%AE%BE%E6%B3%A2%E9%95%BF%E4%B8%BA%CE%BB%EF%BC%8C%E7%A9%BA%E9%97%B4%E5%85%B7%E6%9C%89%CE%B4%E5%88%86%E5%B8%83%E5%87%BD%E6%95%B0%E7%9A%84%E5%85%A5%E5%B0%84%E8%BE%90%E5%B0%84%EF%BC%8C%E4%BB%8E%20%28%CE%B80%EF%BC%8C%CF%860%29%20%E6%96%B9%E5%90%91%EF%BC%8C%E4%BB%A5%E8%BE%90%E5%B0%84%E4%BA%AE%E5%BA%A6L0,%28%CE%B80%EF%BC%8C%CF%860%EF%BC%8C%CE%BB%29%E6%8A%95%E5%B0%84%E5%90%91%E7%82%B9%E7%9B%AE%E6%A0%87%EF%BC%8C%E9%80%A0%E6%88%90%E8%AF%A5%E7%82%B9%E7%9B%AE%E6%A0%87%E7%9A%84%E8%BE%90%E7%85%A7%E5%BA%A6%E5%A2%9E%E9%87%8F%E4%B8%BAdE%20%28%CE%B80%EF%BC%8C%CF%860%EF%BC%8C%CE%BB%29%20%3D%20L0%20%28%CE%B80%EF%BC%8C%CF%860%EF%BC%8C%CE%BB%29%20cos%CE%B80%20d%CE%A9%E3%80%82)）

    本文设计的解码器网络：<img src='https://img-blog.csdnimg.cn/e0504d3786df4b37ad59995d8c678ae5.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAYm51Y3N5,size_20,color_FFFFFF,t_70,g_se,x_16' align='center' width=70%></img>

   &emsp; &emsp;在图中我们可以看到，解码器由三个不同的网络共同构成。$SepNet, DispNet, FusionNet$ 三个网络各自有着对应的任务。同时也有着一些方法，如 $warping$，以及涉及到 $Occlusion \_ map, Disparity \_ map,Feature\_map$`遮挡图是通过特征图得到的么？是的话为什么可以直接得到，是物理方法么？`等的获取，接下来我们将一一介绍。

- $SepNet$：该网络的作用是特征分离，具体地说，这个模块将元通道 $Z$ 分解为一系列特征映射 $\{F_i\}$，其中每个 $F_i$ 包含视图 $i$ 的特征信息，和视觉通道独立，可以和视觉通道结合重建视图 $i$。

- $DispNet$：该网络的作用是视差恢复。具体来说，从特征图 $F_i$ 中提取出 $I_i$ 视差图 $D(u_i)$。接着我们如果给出某一个光场视图$L(u_j)$，我们可以得到合成视图（即warped方法）$\bar{L}(u_i)$：
  $$
  \bar{L}(u_i,x)=L(u_j, x+(u_j-u_i)\times D(u_i,x))
  $$
  &emsp; &emsp;因此，我们可以明确认识到，对于任何一个视差图$D(u_i)$，我们将其与$I_c$进行合成得到 $\bar L(u_i)$。此时我们得到的图理论上和$\widehat L(u_i)$一致（其中$\widehat L=L$，作为重建光场的目标），但实际上由于遮挡和非朗博效应总会出现不同。为了解决这个问题，我们需要另一个网络修复他，但是网络的执行时间可以由 $D(u_i)$ 的遮挡情况自适应地确定，为了度量遮挡程度，我们记 
  $$
  O(u_i, x) = ||D(u_i,x)-D(u_c,x+(u_c-u_i)\times D(u_i,x))||_1
  $$
  &emsp;&emsp;其中$D(u_c)$ 表示 $I_c$ 的视差图。并且有 $O(u_i,x)$越大，接下来的网络需要的时间就越长。

- $FusionNet$：该网络的目的是重建遮挡细节和非朗博效应，这个网络通过大规模的训练达到将视觉图和遮挡图变回光场图像的目的。大规模的训练遇到的关键问题是数据问题，为了解决这个问题，本文提出了一个算子 $G$，算子 $G$ 通过平均采样的方式更改光场，对于原本的IO组$\{\widetilde I_c,\widehat L\}$，我们更改为$\{G( I_c),G(L)\}$，从而得到大规模数据。

4. #### 损失函数
   &emsp; &emsp;网络通过最小化损失函数联合训练编码和解码子网络，本文的损失函数分为三个部分，扭曲一致性损失（也就是$warped$模块） $\zeta_W$，视差正则损失 $\zeta_D$ ，重建损失 $\zeta_R$。故而总损失为：
   $$
   \zeta = \omega_1\zeta_W + \omega_2\zeta_D + \omega_3\zeta_R
   $$
   &emsp; &emsp;在这里，我们设定$\omega_1 = 0.5,\omega_2 = 0.01, \omega_3 = 1.0$。接下来我们将逐一介绍三个损失。

   &emsp; &emsp;$\zeta_W$：扭曲一致性损失。该损失用来度量$warped$之后的光场$\bar L$ 和中央视觉通道 $I_c$之间的差别，使得视差能够更好地被还原。公式写作：
   $$
   \zeta_W = E_{L_i \in S}\{||\bar L_i - L_i||_1\}
   $$
   &emsp;&emsp;`warp是一个物理方法，这个方法由于不考虑遮挡理论上本身就和预期结果不同，如果使用这种损失有没有可能会让视差的生成网络过拟合呢？（也就是视差的生成网络不去生成视差，而是某种能和中央通道直接合成光场的图片【这并不是值得开心的，因为之后还要和遮挡图进行进一步的融合】）`

   &emsp; &emsp;其中，$S$ 即为光场构成的数据集，这个损失即为所有光场的中央视觉通道与对应的$\bar L_i$的一阶范数。该损失限制了从$DispNet$中生成的视差图，使其更加接近于真实视差。

   &emsp; &emsp;但是仅仅由这个损失进行限制仍然是不够的，由于这个损失并不能度量到没有纹理的区域的扭曲视图是否相近，这种失误将会造成遮挡部分无法进行一致性地传播，将会加大 $FusionNet$ 的训练量。为了解决这个问题，我们引入正则损失 $\zeta_D$，加强相邻视图的视差一致性。
   $$
   \zeta_D = E_{L_i \in S}\{||D(u_i,x)-D(u_i-1,x+D(u_i,x))||_1\}
   $$
   &emsp;&emsp;这个损失函数实际上限制了在预测后的光场中，相邻视图需要尽可能地一致，保证相邻视图之间的视差一致性。每个视图在某种程度上都会受到其他视图的影响和制约。

   &emsp; &emsp;在最后，我们通过重建的光场和原本的光场相一致的限制构造重建损耗 $\zeta_R$：
   $$
   \zeta_R = E_{L_i\in S}\{\alpha||\widetilde L_i-\widehat L_i||_1+\beta||SSIM(\widetilde L_i, \widehat L_i)||_1\}
   $$
   &emsp;&emsp;实验中，$SSIM(L_1,L_2)$表示两个光场的所有视角的平均$SSIM$值，[计算见此](https://baike.baidu.com/item/SSIM/2091025)。并且预设$\alpha=1.0,\beta=0.02$。选取$SSIM$度量将在模糊区域获得更精确的细节。

### 实现的细节

1. 数据集
   &emsp; &emsp;本文的数据集来自两个公开数据集，斯坦福和MIT的相机光场数据，总共收集了406个光场，每个光场有14\*14个角度和376\*541个像素。在训练的过程中，随机选取326个作为训练集，剩余80个作为测试集，对每个光场裁取7\*7个角度以及每个角度128\*128个像素组成的照片。对训练集进行了数据增强，即水平翻转或者旋转。通过数据增强，最终得到14447个光场样本进行训练。

2. 训练
   &emsp; &emsp;训练时编码子网和解码子网是联合训练的，但是这种训练方式较慢，本文提出将这个训练分为两个部分。首先是进行除了 $FusionNet$ 之外的网络进行训练，这个阶段迭代10000次，这个训练的目的是加速以及预热$DispNet$ 。在第二阶段，包括 $FusionNet$ 在内 的所有网络进行 50000 次训练。训练中使用Adam优化器，第一阶段学习率为0.0002，第二阶段学习率下降为第一阶段学习率的1%。

   &emsp; &emsp;另外，本文采用pytorch实现了方法，这个实验在两台NVDIA GeForce GTX 1080 Ti 上运行了48个小时。完成的模型对于7\*7的光场输入编码元通道约需要0.04秒，重建光场约需要2秒。

   &emsp; &emsp;源代码和数据尚未开源，等待本文正式发布将会开源代码和数据。

### 实验的结果

&emsp; &emsp;我们可以看到，无论是颜色编辑、颜色转换、阴影去除、风格转换、对比增强等针对2D图片的编辑方法，在本模型的支持下均完成了对于重建光场的一致性传播。
 <img src='https://img-blog.csdnimg.cn/f7655d1624894b87a117e25801a7b14d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAYm51Y3N5,size_20,color_FFFFFF,t_70,g_se,x_16' width=110% align='center'></img>

