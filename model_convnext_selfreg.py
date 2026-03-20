import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# ==============================================================================
# Part 1: RSTD 基础模块 (保持不变)
# ==============================================================================
class RSTD(nn.Module):
    def __init__(self, in_channels, feat_height, feat_width, global_token_dim=64, momentum=0.99):
        super(RSTD, self).__init__()
        self.C, self.H, self.W = in_channels, feat_height, feat_width
        self.D, self.momentum = global_token_dim, momentum
        
        self.freq_h, self.freq_w = feat_height, feat_width // 2 + 1
        self.register_buffer("memory_bank", torch.zeros(1, self.C, self.freq_h, self.freq_w, dtype=torch.complex64))
        
        self.psi_amp = nn.Sequential(nn.Linear(self.D, self.C), nn.ReLU(), nn.Linear(self.C, self.C))
        self.psi_phase = nn.Sequential(nn.Linear(self.D, self.C), nn.ReLU(), nn.Linear(self.C, self.C))
        self.phi_evolve = nn.Sequential(nn.Linear(self.C, self.D), nn.GELU(), nn.Linear(self.D, self.D))
        self.norm_g = nn.LayerNorm(self.D)
        
        self.proj_g_spatial = nn.Conv2d(self.D, self.C, kernel_size=1)
        self.align_conv = nn.Conv2d(self.C * 2, 1, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, x, g):
        B, C, H, W = x.shape
        x_spec = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        amp_mod = torch.tanh(self.psi_amp(g)).view(B, C, 1, 1)
        rect_amp = x_spec.abs() * (1 + self.memory_bank.abs() * amp_mod)
        
        phase_mod = torch.sigmoid(self.psi_phase(g)).view(B, C, 1, 1)
        rect_phase = x_spec.angle() + self.memory_bank.angle() * phase_mod
        
        R = torch.fft.irfft2(torch.polar(rect_amp, rect_phase), s=(H, W), dim=(-2, -1), norm='ortho')
        g_new = self.norm_g(g + self.phi_evolve(torch.mean(R, dim=(2, 3))))
        
        g_exp = self.proj_g_spatial(g_new.view(B, self.D, 1, 1)).expand(-1, -1, H, W)
        m_filter = torch.sigmoid(self.align_conv(torch.cat([x, g_exp], dim=1))) 
        x_final = x + self.gamma * (m_filter * g_exp)
        
        if self.training:
            self.memory_bank = self.momentum * self.memory_bank + (1 - self.momentum) * x_spec.mean(0, keepdim=True).detach()
        return x_final, g_new

# ==============================================================================
# Part 2: ConvNeXt 相关组件 (引入 LayerNorm 和 RSTD_ConvNeXtBlock)
# ==============================================================================
class LayerNorm(nn.Module):
    """ 支持 channels_last (默认) 或 channels_first 的 LayerNorm """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class RSTD_ConvNeXtBlock(nn.Module):
    """
    基于 ConvNeXt 思想的 RSTD 块：
    7x7 Depthwise Conv -> LayerNorm -> 1x1 Conv (扩张) -> GELU -> 1x1 Conv (压缩) -> RSTD
    """
    def __init__(self, dim, h, w, global_dim, mlp_ratio=4, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, mlp_ratio * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_ratio * dim, dim)
        self.rstd = RSTD(dim, h, w, global_token_dim=global_dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)

    def forward(self, x, g):
        input_x = x
        x = self.dwconv(x)
        x = self.norm(x)
        
        # 转换为 channels_last 进行线性层操作
        x = x.permute(0, 2, 3, 1) 
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) 

        # 应用 RSTD
        x, g_new = self.rstd(x, g)
        x = input_x + x
        return x, g_new

# ==============================================================================
# Part 3: 融合版 RSTD_ConvNeXt_SelfReg
# ==============================================================================
class RSTD_ConvNeXt_SelfReg(nn.Module):
    def __init__(self, num_classes=2, input_channels=3, img_size=256, 
                 embed_dims=[32, 64, 128, 256], global_dim=64):
        super().__init__()
        self.num_stages = len(embed_dims)
        self.global_dim = global_dim
        self.init_global_token = nn.Parameter(torch.randn(1, global_dim) * 0.02)
        
        # Stem 改为 ConvNeXt 风格
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, embed_dims[0], kernel_size=4, stride=4) if input_channels == 3 else nn.Conv2d(input_channels, embed_dims[0], kernel_size=3, padding=1),
            LayerNorm(embed_dims[0], data_format="channels_first")
        )
        
        # 这里重置 img_size，因为 stem 如果用了 stride=4，尺寸会缩小
        # 假设这里依然保持你的常规 1x1 级别初始分辨率逻辑，改回普通的 3x3 卷积以防尺寸对不上
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, embed_dims[0], kernel_size=3, padding=1),
            LayerNorm(embed_dims[0], data_format="channels_first"),
            nn.GELU()
        )
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        curr_h, curr_w = img_size, img_size
        
        for i in range(self.num_stages):
            block = RSTD_ConvNeXtBlock(embed_dims[i], curr_h, curr_w, global_dim)
            self.encoder_blocks.append(block)
            
            if i < self.num_stages - 1:
                # Downsample 层也替换为 ConvNeXt 风格的 LayerNorm
                self.downsamples.append(nn.Sequential(
                    LayerNorm(embed_dims[i], data_format="channels_first"),
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=2, stride=2)
                ))
                curr_h //= 2; curr_w //= 2
        
        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(self.num_stages - 2, -1, -1):
            # Upsample 保持双线性插值或反卷积皆可，这里沿用你的双线性
            self.upsamples.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(embed_dims[i+1], embed_dims[i], kernel_size=1),
                LayerNorm(embed_dims[i], data_format="channels_first")
            ))
            
            h_up = img_size // (2 ** i); w_up = img_size // (2 ** i)
            # Fusion 层也改为 LayerNorm + GELU
            fusion = nn.Sequential(
                nn.Conv2d(embed_dims[i]*2, embed_dims[i], kernel_size=1, bias=False),
                LayerNorm(embed_dims[i], data_format="channels_first"),
                nn.GELU()
            )
            block = RSTD_ConvNeXtBlock(embed_dims[i], h_up, w_up, global_dim)
            self.decoder_blocks.append(nn.ModuleList([fusion, block]))
            
        self.final_conv = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)
        
    def forward(self, x):
        B = x.shape[0]
        g = self.init_global_token.expand(B, -1)
        is_train = self.training 
        intermediate_features = []
        
        x = self.stem(x)
        
        # Encoder
        skips = []
        for i in range(self.num_stages):
            x, g = self.encoder_blocks[i](x, g)
            if is_train: intermediate_features.append(x)
            if i < self.num_stages - 1:
                skips.append(x)
                x = self.downsamples[i](x)
        
        # Decoder
        for i, (up, dec_layers) in enumerate(zip(self.upsamples, self.decoder_blocks)):
            fusion_layer, rstd_block = dec_layers
            x = up(x)
            x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = fusion_layer(x)
            x, g = rstd_block(x, g)
            if is_train: intermediate_features.append(x)
            
        out = self.final_conv(x)
        
        if is_train:
            return out, intermediate_features
        return out

# ==============================================================================
# Part 4: SelfRegLoss (已升级为空间注意力一致性 SAC 版本)
# ==============================================================================
class SelfRegLoss(nn.Module):
    def __init__(self, lambda_scr=0.015, lambda_ifd=0.015, main_criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.lambda_scr = lambda_scr
        self.lambda_ifd = lambda_ifd
        self.main_criterion = main_criterion
        self.mse_loss = nn.MSELoss()

    def _random_channel_selection(self, feat, target_channels):
        B, C, H, W = feat.shape
        if C == target_channels: return feat
        indices = torch.randperm(C)[:target_channels].to(feat.device)
        return torch.index_select(feat, 1, indices)

    def forward(self, model_output, target):
        if isinstance(model_output, tuple):
            pred, features = model_output
        else:
            pred, features = model_output, None

        # 1. 计算主干任务的交叉熵损失
        l_main = self.main_criterion(pred, target)

        if features is None:
            return l_main

        teacher = features[-1]
        students = features[:-1]

        # =======================================================
        # 2. CSC: 跨层语义一致性 (Cross-Layer Semantic Consistency)
        # =======================================================
        l_scr = 0.0
        for student in students:
            # 空间维度对齐
            t_down = F.adaptive_avg_pool2d(teacher, student.shape[2:])
            min_C = min(teacher.shape[1], student.shape[1])
            
            # 通道维度随机匹配
            s_aligned = self._random_channel_selection(student, min_C)
            t_aligned = self._random_channel_selection(t_down, min_C)
            

            l_scr += self.mse_loss(s_aligned, t_aligned.detach())
            
        l_scr = l_scr / len(students) if students else 0.0

        # =======================================================
        # 3. IFD: 特征内蒸馏 (Intra-Feature Distillation)
        # =======================================================
        l_ifd = 0.0
        for feat in features:
            B, C, H, W = feat.shape
            if C >= 2:

                channel_norms = torch.norm(feat.view(B, C, -1), p=2, dim=-1) # shape: [B, C]
                # 降序排列获取通道索引
                sorted_indices = torch.argsort(channel_norms, dim=1, descending=True) # shape: [B, C]
                
                half_C = C // 2
                l_ifd_batch = 0.0
                
                # 由于是 Instance-level，必须按 Batch 逐个处理切片
                for b in range(B):
                    idx = sorted_indices[b]
                    strong_idx = idx[:half_C]
                    weak_idx = idx[half_C:2*half_C]
                    
                    F_strong = feat[b:b+1, strong_idx, :, :] # 保持 [1, C/2, H, W] 形状
                    F_weak = feat[b:b+1, weak_idx, :, :]
                    
                    
                    A_strong = torch.sigmoid(torch.mean(F_strong, dim=1, keepdim=True)) # [1, 1, H, W]
                    A_weak = torch.sigmoid(torch.mean(F_weak, dim=1, keepdim=True))     # [1, 1, H, W]
             
                    l_ifd_batch += self.mse_loss(A_weak, A_strong.detach())
                
                l_ifd += l_ifd_batch / B
                
        l_ifd = l_ifd / len(features)

        # 4. 汇总 Total Loss
        total_loss = l_main + self.lambda_scr * l_scr + self.lambda_ifd * l_ifd
        # 为了适配你的训练脚本，按元组返回
        return total_loss, l_main, l_scr, l_ifd

# ==============================================================================
# 测试脚本
# ==============================================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using Device: {device}")
    
    img_size = 56
    model = RSTD_ConvNeXt_SelfReg(img_size=img_size).to(device)
    criterion = SelfRegLoss(main_criterion=nn.CrossEntropyLoss())
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"📦 Model Parameters: {params:.2f} M")
    
    dummy_img = torch.randn(2, 3, img_size, img_size).to(device)
    dummy_gt = torch.randint(0, 2, (2, img_size, img_size)).long().to(device)

    print("\n" + "="*40)
    print("🎯 Training Mode Check")
    model.train()
    output_train = model(dummy_img)
    loss_tuple = criterion(output_train, dummy_gt)
    print("Output Type:", type(output_train))
    print(f"Total Train Loss: {loss_tuple[0].item():.4f}")

    print("\n" + "="*40)
    print("🔬 Eval Mode Check")
    model.eval()
    with torch.no_grad():
        output_eval = model(dummy_img)
        loss_val = criterion(output_eval, dummy_gt)
    print("Output Type:", type(output_eval))
    print(f"Eval Loss: {loss_val.item():.4f}")
    
    print("\n" + "="*40)
    print("⏱️ Running FPS Benchmark...")
    batch_size = dummy_img.shape[0]
    repetitions = 50 
    
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_img) # Warm-up
            
    if torch.cuda.is_available(): torch.cuda.synchronize() 
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(repetitions):
            _ = model(dummy_img)
            if torch.cuda.is_available(): torch.cuda.synchronize() 
                
    total_time = time.time() - start_time
    fps = (batch_size * repetitions) / total_time
    print(f"Throughput: {fps:.2f} FPS (Batch Size: {batch_size})")
    print("="*40)