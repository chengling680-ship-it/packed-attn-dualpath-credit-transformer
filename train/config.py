class OptimalConfig:

    # ===== 数据参数 =====
    seq_len = 21
    c_in = 200
    num_vars = 201
    
    # ===== 核心模型参数 =====
    d_model = 192                   # ⭐ 关键：384（12整除，适中）
    n_heads = 6                      # 384 / 6 = 64 (head_dim)
    e_layers = 3                     # 4层足够（时序短）
    d_ff = 2*d_model                      # 2倍 d_model
    
    # ===== 嵌入层参数 =====
    gru_hidden_dim = 128              # GRU隐藏层
    
    # ===== 正则化参数 =====
    dropout = 0.2
    attention_dropout = 0.1
    path_dropout = 0.1               # Stochastic Depth
    activation = 'gelu'
    use_norm = True                # Pre-norm（更稳定）
    
    
    # ===== 训练参数（针对12GB显存优化）=====
    batch_size = 128                # ⭐ 关键：256
    
    learning_rate = 1e-5
    weight_decay = 1e-4
    max_epochs = 20                 #vertical模型的训练轮数
    early_patience = 6
    scheduler_patience = 3
    scheduler_factor  = 0.7

    #fusion模型
    grad_clip = 1.0
    warmup_epochs = 2
    

    
    



