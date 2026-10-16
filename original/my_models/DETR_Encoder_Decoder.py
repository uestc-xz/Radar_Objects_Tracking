import torch.nn as nn
import torch,math
class EncoderLayer(nn.Module):
    def __init__(self, attention_dim=128, attention_heads=4):
        super().__init__()
        self.attention_dim = attention_dim
        self.attention_heads = attention_heads
        self.head_dim = attention_dim // attention_heads
        # 自注意力相关层
        self.qkv_proj = nn.Linear(attention_dim, attention_dim * 3)
        self.out_proj = nn.Linear(attention_dim, attention_dim)
        # self.rel_pos_bias = nn.Parameter(torch.randn(attention_heads, 36, 36))
        self.ffn = nn.Sequential(
            nn.Linear(attention_dim, attention_dim * 4),
            nn.ReLU(),
            nn.Linear(attention_dim * 4, attention_dim)
        )

        # xy 归一化
        self.norm1 = nn.LayerNorm(attention_dim)
        self.norm2 = nn.LayerNorm(attention_dim)
    def forward(self, x):
        # x形状: [B, T, dim]
        batch_size, time_steps, _ = x.shape
        qkv = self.qkv_proj(x)
        # print(f"qkvout shape{qkv.shape}")
        qkv = qkv.reshape(batch_size, time_steps, 3, self.attention_heads, self.head_dim)
        # print(f"qkvout shape after reshape{qkv.shape}")
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # print(f"q shape {q.shape}")
        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # attn_weights = attn_weights + self.rel_pos_bias[:, :time_steps, :time_steps].unsqueeze(0)

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, time_steps, self.attention_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.norm1(x + attn_output)
        ffn_output = self.ffn(attn_output)
        output = self.norm2(attn_output + ffn_output)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model=128, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        #掩码自注意（用于查询向量之间的交互）
        self.self_attn_qkv = nn.Linear(d_model, d_model * 3)  # QKV投影
        self.self_attn_out = nn.Linear(d_model, d_model)  # 输出投影
        self.norm1 = nn.LayerNorm(d_model)

        #交叉注意力查询Encoder输出
        self.cross_attn_q = nn.Linear(d_model, d_model)  # Q投影（来自解码器）
        self.cross_attn_kv = nn.Linear(d_model, d_model * 2)  # KV投影（来自编码器）
        self.cross_attn_out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)

        #前馈网络FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)  # 归一化

    def forward(self, dec_input, enc_output):
        # dec_input: 解码器输入 [B, num_queries, d_model]
        # enc_output: 编码器输出 [B, T, d_model]
        batch_size, num_queries, _ = dec_input.shape
        _, time_steps, _ = enc_output.shape

        qkv = self.self_attn_qkv(dec_input)  # [B, num_queries, 3*d_model]
        qkv = qkv.reshape(batch_size, num_queries, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        # 转置为 [B, num_heads, num_queries, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        mask = torch.triu(torch.ones(num_queries, num_queries), diagonal=1).bool()  #[num_queries, num_queries]
        mask = mask.to(dec_input.device)  # 移到相同设备

        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  #[B, num_heads, num_queries, num_queries]
        attn_weights = attn_weights.masked_fill(mask, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = attn_weights @ v  #[B, num_heads, num_queries, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  #[B, num_queries, num_heads, head_dim]
        attn_output = attn_output.reshape(batch_size, num_queries, self.d_model)  # 合并多头
        attn_output = self.self_attn_out(attn_output)  # 输出投影

        self_attn_out = self.norm1(dec_input + attn_output)  # [B, num_queries, d_model]

        #交叉（Encoder-Decoder Attention）

        q = self.cross_attn_q(self_attn_out)  # [B, num_queries, d_model]
        kv = self.cross_attn_kv(enc_output)  # [B, T, 2*d_model]
        # 拆分多头
        q = q.reshape(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)  #[B, num_heads, num_queries, head_dim]
        kv = kv.reshape(batch_size, time_steps, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)  # [B, T, num_heads, head_dim] → 转置后 [B, num_heads, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cross_attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, num_heads, num_queries, T]
        cross_attn_weights = torch.softmax(cross_attn_weights, dim=-1)

        cross_attn_output = cross_attn_weights @ v  # [B, num_heads, num_queries, head_dim]
        cross_attn_output = cross_attn_output.transpose(1, 2).contiguous()  # [B, num_queries, num_heads, head_dim]
        cross_attn_output = cross_attn_output.reshape(batch_size, num_queries, self.d_model)  # 合并多头
        cross_attn_output = self.cross_attn_out(cross_attn_output)  # 输出投影

        cross_attn_out = self.norm2(self_attn_out + cross_attn_output)  # [B, num_queries, d_model]

        ffn_output = self.ffn(cross_attn_out)
        ffn_out = self.norm3(cross_attn_out + ffn_output)  # [B, num_queries, d_model]

        return ffn_out
    # def forward(self, dec_input, enc_output):
    #     # dec_input: 解码器输入 [B, num_queries, d_model]
    #     # enc_output: 编码器输出 [B, T, d_model]
    #     batch_size, num_queries, _ = dec_input.shape
    #     _, time_steps, _ = enc_output.shape
    #
    #     # 1. 自注意力 (Self-Attention) - 保持不变
    #     qkv = self.self_attn_qkv(dec_input)
    #     qkv = qkv.reshape(batch_size, num_queries, 3, self.num_heads, self.head_dim)
    #     q, k, v = qkv.unbind(2)
    #     q = q.transpose(1, 2)
    #     k = k.transpose(1, 2)
    #     v = v.transpose(1, 2)
    #     mask = torch.triu(torch.ones(num_queries, num_queries), diagonal=1).bool().to(dec_input.device)
    #
    #     attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    #     attn_weights = attn_weights.masked_fill(mask, -1e9)
    #     attn_weights = torch.softmax(attn_weights, dim=-1)
    #
    #     attn_output = attn_weights @ v
    #     attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, num_queries, self.d_model)
    #     attn_output = self.self_attn_out(attn_output)
    #     self_attn_out = self.norm1(dec_input + attn_output)
    #
    #     # 2. 交叉注意力 (Cross-Attention)
    #     q = self.cross_attn_q(self_attn_out)
    #     kv = self.cross_attn_kv(enc_output)
    #
    #     q = q.reshape(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
    #     kv = kv.reshape(batch_size, time_steps, 2, self.num_heads, self.head_dim)
    #     k, v = kv.unbind(2)
    #     k = k.transpose(1, 2)
    #     v = v.transpose(1, 2)
    #
    #     # 计算注意力权重 [B, num_heads, num_queries, T]
    #     cross_attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    #     cross_attn_weights = torch.softmax(cross_attn_weights, dim=-1)  # <--- 我们需要的权重张量
    #
    #     cross_attn_output = cross_attn_weights @ v
    #     cross_attn_output = cross_attn_output.transpose(1, 2).contiguous().reshape(batch_size, num_queries,
    #                                                                                self.d_model)
    #     cross_attn_output = self.cross_attn_out(cross_attn_output)
    #
    #     cross_attn_out = self.norm2(self_attn_out + cross_attn_output)  # [B, num_queries, d_model]
    #
    #     # 3. 前馈网络 (FFN)
    #     ffn_output = self.ffn(cross_attn_out)
    #     ffn_out = self.norm3(cross_attn_out + ffn_output)  # [B, num_queries, d_model]
    #
    #     # 【修改 1】：将多头注意力权重平均到单头，形状变为 [B, num_queries, T]
    #     # 这一步是为了方便后续的单图可视化（48个特征点）
    #     mean_cross_attn_weights = cross_attn_weights.mean(dim=1)
    #
    #     # 【修改 2】：返回两个值：最终输出和平均注意力权重
    #     return ffn_out, mean_cross_attn_weights

class Decoder(nn.Module):
    def __init__(self, d_model=128, num_queries=1, num_layers=2):
        super().__init__()
        self.num_queries = num_queries  # 最多预测5个目标
        self.d_model = d_model

        # 1. 可学习的目标查询向量
        self.object_queries = nn.Parameter(torch.randn(num_queries, d_model))  # [5, 128]

        # nn.init.xavier_uniform_(self.object_queries) ## 初始化方法

        # 2. 查询向量的位置编码（区分不同查询）
        self.query_pos_enc = self._create_query_pos_enc(num_queries, d_model)

        # 3. 堆叠多层解码器
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, num_heads=4)
            for _ in range(num_layers)
        ])

        # 4. 输出预测头x（将特征映射为坐标）
        self.coord_headx = nn.Sequential(
            nn.Linear(d_model, 2*d_model),nn.ReLU(),
            nn.Linear(2*d_model, 4 * d_model), nn.ReLU(),
            nn.Linear(4 * d_model, 2 * d_model), nn.ReLU(),
            nn.Linear(2*d_model, d_model),nn.ReLU(),
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 1)  # 输出(x, y)坐标
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 2*d_model),nn.ReLU(),
            nn.Linear(2*d_model, 4 * d_model), nn.ReLU(),
            nn.Linear(4 * d_model, 2 * d_model), nn.ReLU(),
            nn.Linear(2*d_model, d_model),nn.ReLU(),
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 2)  # 输出(x, y)坐标
        )
        self.coord_heady = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.ReLU(),
            nn.Linear(2 * d_model, 4 * d_model), nn.ReLU(),
            nn.Linear(4 * d_model, 2 * d_model), nn.ReLU(),
            nn.Linear(2 * d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 1)  # 输出(x, y)坐标
        )
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.ReLU(),
            nn.Linear(2 * d_model, 4 * d_model), nn.ReLU(),
            nn.Linear(4 * d_model, 2 * d_model), nn.ReLU(),
            nn.Linear(2 * d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 2)  # 输出置信度
        )
    def _create_query_pos_enc(self, num_queries, d_model):
        """为查询向量生成正弦余弦位置编码"""
        position = torch.arange(num_queries, dtype=torch.float32).unsqueeze(1)  # [num_queries, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [d_model//2]

        pe = torch.zeros(num_queries, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [num_queries, d_model]

    def forward(self, enc_output):
        """
        enc_output: 编码器输出 [B, T, d_model]
        return: 预测的坐标 [B, num_queries, 2]
        """
        batch_size = enc_output.shape[0]
        # 扩展查询向量到批次维度
        queries = self.object_queries.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, num_queries, d_model]
        # print("扩展后queries形状:", queries.shape)  # 预期：[B,1,128]（B是批次大小，如48）
        # 添加位置编码
        queries = queries + self.query_pos_enc.to(queries.device)  # [B, num_queries, d_model]
        layer_outputs = []
        # 逐层通过解码器
        dec_output = queries
        # print(dec_output.shape)
        for layer in self.decoder_layers:
            dec_output = layer(dec_input=dec_output, enc_output=enc_output)
            # print(f"dec_out : {dec_output.shape}")
            layer_outputs.append(dec_output)
        # 生成坐标预测
        # pred_coords = self.coord_head(dec_output)  # [B, num_queries, 2]
        # layer_preds = [self.coord_head(out) for out in layer_outputs]
        layer_preds = [torch.cat([self.coord_headx(out), self.coord_heady(out)], dim=1) for out in layer_outputs]
        layer_class_preds = [self.classifier(out) for out in layer_outputs]
        layer_conf_preds = [self.conf_head(out) for out in layer_outputs]
        # print("每层layer_pred形状:", [p.shape for p in layer_preds])
        return layer_preds,layer_class_preds,layer_conf_preds,dec_output
    # def forward(self, enc_output):
    #     """
    #     enc_output: 编码器输出 [B, T, d_model]
    #     return: 预测的坐标 [B, num_queries, 2]
    #     """
    #     batch_size = enc_output.shape[0]
    #     # 扩展查询向量到批次维度
    #     queries = self.object_queries.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, num_queries, d_model]
    #     # 添加位置编码
    #     queries = queries + self.query_pos_enc.to(queries.device)  # [B, num_queries, d_model]
    #
    #     layer_outputs = []
    #     cross_attn_weights = []  # <--- 新增：用于收集每层权重
    #
    #     # 逐层通过解码器
    #     dec_output = queries
    #
    #     # 假设 DecoderLayer 现在返回 (dec_output, attn_weights)
    #     for layer in self.decoder_layers:
    #         dec_output, attn_weights = layer(dec_input=dec_output, enc_output=enc_output)  # <--- 修改：接收两个返回值
    #
    #         layer_outputs.append(dec_output)
    #         cross_attn_weights.append(attn_weights)  # <--- 新增：收集权重
    #
    #     # 生成坐标预测 (保持不变)
    #     layer_preds = [torch.cat([self.coord_headx(out), self.coord_heady(out)], dim=1) for out in layer_outputs]
    #     layer_class_preds = [self.classifier(out) for out in layer_outputs]
    #     layer_conf_preds = [self.conf_head(out) for out in layer_outputs]
    #
    #     # 【关键修改】：返回 5 个值
    #     return layer_preds, layer_class_preds, layer_conf_preds, dec_output, cross_attn_weights