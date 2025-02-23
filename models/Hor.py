import torch
import torch.nn as nn

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)


class gnconv(nn.Module):
    def __init__(self, dim, order=3, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order# 空间交互的阶数，即n
        self.dims = [dim // 2 ** i for i in range(order)]# 将2C在不同阶的空间上进行切分，对应公式3.2
        self.dims.reverse()# 反序，使低层通道少，高层通道多
        self.proj_in = nn.Conv2d(dim, 2* dim, 1)# 输入x的线性映射层，对应$\phi(in)$

        if gflayer is None:# 是否使用Global Filter
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:# 在全特征上进行卷积，多在后期使用
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)# 输出y的线性映射层，对应$\phi(out)$

        self.pws = nn.ModuleList(# 高阶空间交互过程中使用的卷积模块，对应公式3.4
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s# 缩放系数，对应公式3.3中的$\alpha$
        print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        print(self.dims)

        fused_x = self.proj_in(x)# channel double
        print("fused_x:",fused_x.shape)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)# split channel to c/2**order and c(1-2**oder)
        print("pwa:{}  abc:{}".format(pwa.shape,abc.shape))

        dw_abc = self.dwconv(abc) * self.scale
        print('dw_abc:{}'.format(dw_abc.shape))

        dw_list = torch.split(dw_abc, self.dims, dim=1)

        # 两个相同尺寸的张量相乘，对应元素的相乘就是这个哈达玛积(mul)
        x = pwa * dw_list[0]
        print("x:{}".format(x.shape))

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
            print('conv[{}]:{} * dw_list[{}]:{} = x:{}'.format(i,self.pws[i],
                                                               i+1,dw_list[i+1].shape,
                                                               x.shape))

        x = self.proj_out(x)

        return x

if __name__ == '__main__':
    x=torch.randn((2,64,20,20))
    gn=gnconv(64)
    out=gn(x)
    print(out.shape)
