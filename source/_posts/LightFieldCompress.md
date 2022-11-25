---
title: 毕设进度记录
categories: [paper, mypaper]
mathjax: true
date: 2022-08-13 09:31:37
---

## <center>基于单张RGB图像的光场表征</center>

&emsp;&emsp;本文是为了完成毕业论文进行的文献收集和阅读，在此对各个具备较高参考性的文章进行总结提炼，本文最主要参考的文章进行了精读，[链接见此](http://bnucsy.gitee.io/2022/05/02/A%20Learned%20Compact%20and%20Editable%20Light%20Field%20Representation/)。初步的设想下，本文的自编码器分为编码器和解码器两个子网络，编码器把一个光场输入编码为一个RGB图像，光场的结构性信息理论上被隐式地编码到RGB图像之中。而解码器则利用这个图像进行光场重建。

<!-- more -->

&emsp;&emsp;事实上，已经有能够只从一个RGB图中提取信息重建光场的工作，作为自编码器，由于RGB图中存在着光场结构，至少效果要优于这篇[从一张RGB图重建光场](https://arxiv.org/pdf/1708.03292.pdf)的文章。为了达到这个效果，我认为解码器需要能够分离出RGB特征，我将参考【1】中的 $SepNet$ 。

### `2021/11/19 - 2021/11/26`

`参考文献【2】`

&emsp;&emsp;这篇文献中对如何利用光场 $L$ 的 $L(x,u_{0})$ 还原出整个光场 $L$ 提出了一个可行的方法，该方法是一个基于物理学模型和CNN网络的学习结构，具体如下：

&emsp;&emsp;记模型为函数 $f$，有$\widehat L(x,u)=f(L(x,u_0))$。本文将这个函数分为了三个部分，函数 $d(·)$ 用来估计深度图，函数 $r(·)$ 用来使用深度图和光场的第一张图用来近似估计光场 $\widehat L_r$，最后使用函数 $o(·)$ 来估计遮挡和非朗博效应。
$$
D(x,u) = d(L(x,u_0))\\
L_r(x,u) = r(D(x,u),L(x,u_0))\\
\widehat L(x,u)=o(L_r(x,u),D(x,u))
$$
&emsp;&emsp;这三个部分逐个构成了本文的结构，接下来将逐个介绍这三个模块。

1. 深度图预测估计函数 $d$

   &emsp;&emsp;函数 $d$ 是一个基于CNN的网络结构，我们将函数 $d$ 和函数 $o$ 的关键损失（一致性损失）进行联合计算，联合计算的原因是防止 $o$ 过拟合从而导致 $d$ 欠拟合。联合训练误差表示为：
   $$
   \min_{\theta _d, \theta _0}\Sigma _S [||L_r-L||_1+||\widehat L - L||_1+\lambda_c \psi _c(D)+\lambda_{tv}\psi_{tv}(D)]
   $$
   &emsp;&emsp;即，求解使得在数据集 $S$ 中所有的数据误差最小的 $\theta_d$ 和 $\theta_o$ ，这两个参数分别是深度估计网络和遮挡预测网络的参数`（具体是什么参数？）`。$\psi_c,\psi_{tv}$ 是一致性的正则化损失，用来限制深度图的光线角度一致性以及鼓励网络参数的稀疏性。他们分别表示为：
   $$
   \psi_c(D(x,u))=||D(x,u)-D(x+D(x,u),u-1)||_1 \\ \psi_{tv}(D(x,u))=||\nabla_xD(x,u)||_1
   $$

2. 光场预还原函数 $r$ 

   &emsp;&emsp;光场预测函数是一个基于物理的函数，和参考文献【1】的 $warp$ 部分很接近，具体的表达式为：$L_r(x,u)=L(x+uD(x,u),0)$，其中 $D(x,u)$ 就是之前预测得到的深度，本公式相当于使用深度图当做光场的光线入射角度，根据每个点的截面变化为固定值得出近似光场。

3. 光场遮挡修正函数 $o$ 

   &emsp;&emsp;关于遮挡修正，我们实际上在函数 $d$ 中已经对其进行了限制，将之前的损失函数后向传播就可以完成函数 $o$ 的训练。值得一提的是，本函数的网络结构并不是将近似的光场 $L_r$ 直接变为 $\widehat L$，而是学习一个残差块 $\widetilde o$ ，这将更加保证学习到的网络具备填补遮挡的预测效果。其中 $\widetilde o$ 表示为：
   $$
   o(L_r(x,u),D(x,u))[即 \widehat L]=\widetilde o(L_r(x,u),D(x,u)) +L_r(x,u)
   $$
   

$~~~~$关于这几个函数预测网络的具体结构，这些网络均为普通的CNN结构，具体结构存在于支撑文件和源代码之中，我并没有找到。但是据本文作者所说，他在网络结构上并没有太大的创新，只是在网络的激活函数之中，本文发现使用 Tanh 要优于 ELU。

`关于毕设的一些讨论`
&emsp;在指导老师指导下进一步明确了毕设的内容：
&emsp;&emsp;（开题用）motivation —— 1.当前的光场数据压缩算法大多压缩结果不具有直观性，因此想要做出来一个使用和光场中心视图相似来约束的自编码器，中间层即为压缩结果。 2.这种可视化的压缩能够帮助人们认知原光场结构，并且和图片相似的结构支持基于图片的压缩（如JPEG），而这个特性能够弥补最初可能产生的压缩比不足的缺陷。
&emsp;&emsp;（baseline）method —— 1.数据上采用[这篇文章](https://github.com/YingqianWang/LF-DFnet)的数据集，网络上首先使用参考文献【1】的编码器作为编码和解码网络，同时结合 UNet 完成最初版本网络设计，然后再将网络结构逐渐复杂化，预想中我们将会使用[这篇文章](https://github.com/YingqianWang/LF-DFnet)的网络结构。 2.完成这些之后我们会进行更加精细的更改，如增加PNSR和SSIM损失来优化小型区域的分辨率问题，增加可逆的量化损失让网络具备更好的结构。
&emsp;&emsp;（后话）paper —— 1.在论文的编写过程中需要多评价分析现有的压缩算法，比较各自的优劣，尤其强调自己的亮点。 2.对每个视角进行统计，全方位地估计自己方法的不足。 3.除去压缩比，本方法的压缩优势还包括了可以支持一些后续工作，如进行三维重建以及重聚焦。
&emsp;&emsp;（当务之急）todolist —— `1.从 LF-DFnet 论文中下载光场相关数据`，`2.阅读参考文献【1】的代码部分，尤其是编码器以及重写的 dataset 类`，`3.多查询博客以及2012-2017年间的经典网络结构论文，这些资源将会对“为什么要”和“为什么能”的问题提供帮助，如：`[UNet和FCN](https://www.i4k.xyz/article/qq_43703185/105060277)、[深度学习的基本原理1](http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/)、[深度学习的基本原理2](http://neuralnetworksanddeeplearning.com/)

### `2021/11/26-2021/12/3`

&emsp;&emsp;很惭愧，这周我买了两个游戏——传送门1和传送门2，总共肝了20+小时通关了这两个游戏，所以这周几乎没有做任何有意义的工作。唯一有意义的是我在大风天去爬了香山逛了植物园，让我的右膝盖疼了两天，这件事实际上意义重大，我决心老老实实开启养老模式。早睡早起按时从实验室滚蛋，每天至少喝两杯水，每坐一小时要站起来走十分钟等等。
&emsp;&emsp;不过我确实读完了 Unet 网络的论文，只是还没来得及写总结。
&emsp;&emsp;以及完成了一下当时老师布置的任务，包括：①下载数据`这数据也50G+，开玩笑呢，所以这个任务莫得完成`，②参考文献【1】的 Encoder 层试着设计一个初步的网络出来`这个我倒是完成了（刚完成）`，③深度学习基本原理`这个得慢慢来对吧，我暂时没有太看也可以理解对吧`（但是还是明白了一些东西）

&emsp;&emsp;任务②：本论文的网络结构由两个Unet网络组成，两个Unet具有完全相反的特性，因此在此处只给出初版的 encoder 网络（这个网络将 49 个 128\*128\*3 的光场照片提取特征，并最终映射到 128\*128\*1 的空间之中，后续将会设计一个损失函数，使用中央视觉图来限制中间层）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/f8fd481d205e446dac7dbad06e9a0af5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAYm51Y3N5,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
&emsp;&emsp;任务③：大致了解了这些网络设计原理（持续增加）

 - 提取特征（降采样）之后网络的通道数一般会增加，这是因为我们希望提取特征的同时尽量不损失信息。
 - 在各个降采样之前（或之后）我们需要进行几次卷积，这是因为卷积层越深，其非线性可分性就越强，但是过深的网络会造成运算量和内存消耗太大以及过拟合的问题，通常使用两层卷积（像是个传统？）
 - 在卷积之后往往我们需要使用一个线性激活函数，这类函数最初是 $sigmod, tanh$ ，设计他的原因是：①将输出映射到一个特定的区间里便于计算，②把纯线性的连接转化为非线性的连接（这个更重要）。但是这样的激活函数计算过于复杂，这将导致在后向传播的时候运算较慢，因此现在用 $RELU$ 的比较多，这个网络结构就是如此，只在最后一个输出映射的时候使用 $tanh$。
 - 有些卷积层是 $kernel\_size = stride = 1, padding = 0$ 的卷积，这种卷积看似没有进行卷积操作，实际上对不同的通道进行了运算，往往在最后需要调整通道数的时候使用这种卷积层，如本网络最后一层。
 -  可变形卷积：相当于对卷积核的采样点进行了偏移，对于 $b\times h\times w\times c$ 的特征图 $U$，添加一个卷积核，通过 $padding=same$ 输出同样的大小，输出后的 $size=b\times h\times w\times 2c$，称为 offset，offset 的两个通道分别记录了该像素点向水平和竖直方向的偏移量。接着我们把 offset 和 $U$ 进行合并，得到代表绝对坐标的 $V（size=b\times h\times w\times 2c）$，如此我们再对 $U$ 进行普通卷积，卷积时对于 $U$ 中的某一个像素点，我们寻找其在 $V$ 作用下的偏移坐标，往往得到的是一个浮点数。由于此位置并没有实际像素点，我们还需要对其临近的四个实际像素点进行双线性插值得到该像素点的数值。可以看出可变形卷积具备更多的参数和更好的注意力机制，因此往往我们在整个网络的最后 3-4 层使用该卷积。

`PS. 接下来的任务`：
&emsp;&emsp;①完成Unet论文的总结
&emsp;&emsp;②再读几篇经典论文（VGG,ResNet）
&emsp;&emsp;③试着设计一下网络的损失函数

### `2021/12/3-2021/12/27`

`文献【4】—— Unet 网络`

&emsp;&emsp;Unet 提出了一种可以使用较少的数据进行数据增强和端到端训练图像的方法，这个方法最初用于进行细胞分割。后来人们发现这种具有特征提取结构和对称的扩展结构的网络能够在各个应用场景下表现突出，因此Unet就成为了一种经典的网络结构。
&emsp;&emsp;Unet 的相关工作主要体现在细胞分割上，因此这里不介绍它的训练策略（主要是分割的学习方式），仅关注其网络结构：

<center><img src='https://img-blog.csdnimg.cn/78ab4c0af28d4a7ea4f380a529f9db2e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAYm51Y3N5,size_20,color_FFFFFF,t_70,g_se,x_16' width=80%></img></center>

啊这周已经过去了啊，现在已经是12/12号了，任务呢，一个也没有完成，Unet倒是看完了，其他的论文都没怎么读。这周倒也不算浑浑噩噩，把实验的代码写了个大差不差，基本架构倒是都出来了，跑了个马马虎虎的结果。把结果放一下：

原光场的中央视图：![在这里插入图片描述](https://img-blog.csdnimg.cn/c9b318035f95463580cbf7c1eb2b8045.jpg)  &emsp;&emsp;编码器的输出：![请添加图片描述](https://img-blog.csdnimg.cn/cc5e399949834ca89c1880207a5355ba.jpg)&emsp;&emsp;解码器的输出:<img src='https://img-blog.csdnimg.cn/16d8cc2fa919496c8ac736427a355870.jpg?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAYm51Y3N5,size_20,color_FFFFFF,t_70,g_se,x_16' width=24%></img>
&emsp;&emsp;问题还是十分明显的：①压缩后的图像精读并不够，但是这一个差得不是太多。②最关键的是这光场，他不会动啊！
&emsp;&emsp;这个问题大概是因为我设计的损失函数极其简单：$$ \zeta=0.5*||L-L_r||_2 + ||C-L_c||_2$$$~~~~$对，就是这么简单。简单的后果就是，自编码器学会了单纯地把中央视觉图复制到光场的每一个角度分辨率上，这样虽然跟光场没啥关系了，但是让我这个损失在一定程度上变得更小了，确实是一个局部的最优。
&emsp;&emsp;因此现在我需要思考如何设计损失才能限制住重建之后的光场长得像一个光场，而不是81个一模一样的图片。

&emsp;&emsp;在此之后，我又设想出来了一种新的结构：ETDnet，这种结构是将有中间层限制的自编码器的中间层和结果分别加一个判别器，所以这个结构看起来也像两个GAN。网络结构图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/0d0954d88647447aa2aa3f40b9eb1583.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAYm51Y3N5,size_20,color_FFFFFF,t_70,g_se,x_16)
&emsp;&emsp;其中 $E$ 和 $D$ 分别是两个 $U-net$，作为编码器和解码器， $L\_D$,$F\_D$ 分别是两个基于 $DCGAN$ 的判别器，用来和自编码器进行对抗。这个还只是初步的设想，目前跑出来了不加判别器的结果，应用于图像重上色任务：
![在这里插入图片描述](https://img-blog.csdnimg.cn/243f53761e19473ea905d9044df9d03b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAYm51Y3N5,size_20,color_FFFFFF,t_70,g_se,x_16)*从左至右依次为：原图、压缩后灰度图、重上色图*

&emsp;&emsp;事实上如果仔细观察的话，可以看出灰度图的中间有着一些竖着的条纹，这些条纹理论上就是隐编码，但是这并不是我们想要的，因此需要加入一个判别器和编码器进行对抗，使编码器能够更好地生成灰度图。然而最难的还是调参，判别器很快就可以训练得非常好，这样就会导致自编码器直接摆烂，目前我还在艰难地调学习率。

&emsp;&emsp;说回光场的任务：关于压缩后中央视图色调不一致的问题，加入量化损失之后即可解决，关于恢复之后不一致的问题，我尝试了：①使用更大的数据集，这个数据集是灰度数据，角度分辨率为5*5，空间分辨率为64\*64，共有40000+个光场。②损失函数中加入视差图的学习。③调整损失权重，直至只限制光场重建损失。
&emsp;&emsp;遗憾的是，上述三个结果都没有达到预期，计算得到的PSNR值为33+，数据上看起来并不差，但是一旦可视化，模糊和晃动就很明显。目前我获取到了一个更大的彩色数据集，我将利用这个彩色数据集进行进一步的训练。

### `2021/12/27 - 2022/1/1`

&emsp;&emsp;接下来的四天里面，我将要提交开题报告，因此我需要阅读 $LF\_DFnet$ 的论文，这理论上将作为最终的网络结构，因此需要在目录中提及。
&emsp;&emsp;经过四天的讨论和反复修改，目前开题报告已经完整完成了，开题报告中，本文将改名为 “基于单张RGB图像的光场表征” 完成具备JPEG压缩鲁棒性的光场RGB表征和重建。本文初步定下来的不含有 $LF\_DFnet$ 部分，直接使用基于 $Unet$ 的自编码器。接下来就是把代码做出来了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/1b7b77a649f842818947cc9e2282d082.png)

### `2022/1/1 - 2022/1/19`

&emsp;&emsp;啊首先经过了两个星期的期末考试【因为自认为能学会文科辅修了新闻传播学 : ) 】
&emsp;&emsp;经过不懈的努力（划水），实验终于有了进展，润回家后被隔离的第一天结果跑了出来，PSNR很神奇地达到了36+，接着我限制了中央视图，目前的PSNR已经可以交差了，光场PSNR达到35+，中央视图表征PSNR达到42+。

 - 原光场（大小：11.3MB）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/0deee5693b514fb2b698fc87852c8d08.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAYm51Y3N5,size_20,color_FFFFFF,t_70,g_se,x_16)
 - 压缩后RGB表征（大小：250KB）
![在这里插入图片描述](https://img-blog.csdnimg.cn/caa3e7a00ac4467387ae82c68083cec0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAYm51Y3N5,size_20,color_FFFFFF,t_70,g_se,x_16)
 - 重建后光场（大小：10.9MB）
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/b754cc8d9acf47a0a95cd4a907484c75.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAYm51Y3N5,size_20,color_FFFFFF,t_70,g_se,x_16)

&emsp;&emsp;效果非常之好，表征视图很清晰，并且重建视图很标准地还原了原光场。接下来的任务就在于能够实现JPEG压缩，这个实际很简单，就是在中间加一个JPEG压缩层，然后重建，为了能够重建出较好的结果，网络自然会把光场隐式编码到不会被JPEG压缩破坏的地方。
&emsp;&emsp;按照指导老师的安排，现在着手写最终的论文和实验时间尚早，因此接下来我将认真研读 $LF\_DFnet$ 论文和代码，期望能够在一周之内完成论文的精读。

&emsp;&emsp;刚刚毕设开题报告审查反馈，反馈意见居然是题目过于宽泛。。。。。。
&emsp;&emsp;然而唯独题目是老师说我原本的题目（本文标题）太啰嗦又太简单才更改成现在的标题的。不过也完全可以理解，毕竟本科毕设、中文普刊、ICIP、EBCV、CVPR、SIGGRAPH各有各的标准，按照老师的意见、只答不辩毕了业了就行了。

### `2022/1/20-2022/3/28`

&emsp;&emsp;实际上现在已经5月初了，之所以现在才补全，主要是因为在3月份完成终稿及答辩PPT讲稿之后就再也没有继续写的欲望了，直到昨天想要尝试搭建一个自己的blog，今天完成之后才有心思继续写这一篇，总结全文，这篇记录就结束了。

&emsp;&emsp;自从1-20日到2-20日左右，应该是寒假时间的快乐时光，虽然在家里被隔离了半个月，但和对象一起住的日子毕竟还是更快乐一些。简单来说就是，啥也妹干。至于开题反馈，最终我在题目中加了两个字，遂通过。

&emsp;&emsp;从二月底返校之后，我开始快速完成JPEG编码层的搭建，虽然过程中遇到了一些问题：包括pytorch更新的问题、超参数的调优问题、训练的不拟合问题等等。最终我通过一些网络教程手动更新了pytorch，完成了代码的适配。在运行过程中调优超参数我采用同时搜索多个超参数的方式解决，最终结果又上了一个dB。

&emsp;&emsp;这之中实际遇到的问题很多，包括最终的实验也遇到了很多问题，除此之外我又参考了很多篇文献，最终形成了终稿，[这里](https://bnucsy.gitee.io/2022/05/02/represent-light-field-with-a-single-image/)是对终稿的简单概括。本文最终没有采用可变形卷积方法，得到了一个还可以的结果，创新主要集中于思想和任务，预计在2023年初投稿ICPR。

主要参考文献：


 1. [A Learned Compact and Editable Light Field Representation](https://arxiv.org/abs/2103.11314) （一种紧凑的可编辑光场表示）
 2. [Learning to Synthesize a 4D RGBD Light Field from a Single Image](https://arxiv.org/pdf/1708.03292.pdf)（使用单个RGB图像重建光场）
 3. [Light Field Image Super-Resolution Using
Deformable Convolution](https://arxiv.org/abs/2007.03535)（光场超分辨率重建）
 4. [U-Net: Convolutional Networks for Biomedical
Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)（经典论文：Unet网络）