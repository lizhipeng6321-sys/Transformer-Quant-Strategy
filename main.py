import tushare as ts
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
class Config:
    
    tushare_token = 'your token here' 

    # 标的与时间
    stock_code = '000300.SH'  
    start_date = '20190101'
    end_date = '20241231'

    # 模型参数 (可根据需要微调)
    feature_dim = 6       # 因子数量 (对应下面构造的6个因子)
    d_model = 64          # Transformer 内部维度
    nhead = 4             # 多头注意力头数
    num_layers = 2        # Transformer 层数
    seq_len = 20          # 回看过去20个交易日
    dropout = 0.1
    
    # 训练参数
    batch_size = 64
    learning_rate = 0.001
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 数据获取与因子工程 (Data ETL)
# ==========================================
def get_data(config):
    print(f"正在从 Tushare 下载 {config.stock_code} 数据...")
    ts.set_token(config.tushare_token)
    pro = ts.pro_api()
    
    # 获取数据
    # 如果是指数，使用 index_daily; 如果是个股，使用 daily
    try:
        if config.stock_code.endswith('.SH') or config.stock_code.endswith('.SZ'):
             # 尝试作为指数获取
            df = pro.index_daily(ts_code=config.stock_code, start_date=config.start_date, end_date=config.end_date)
            if df.empty: # 如果指数没取到，尝试作为个股
                df = pro.daily(ts_code=config.stock_code, start_date=config.start_date, end_date=config.end_date)
    except Exception as e:
        print(f"数据下载出错: {e}")
        return None

    # Tushare 数据默认是倒序的，必须反转为正序（时间从旧到新）
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    # --- 因子构造 (Feature Engineering) ---
    # 1. 简单收益率
    df['pct_chg'] = df['close'].pct_change()
    
    # 2. 5日波动率 (Rolling Volatility)
    df['vol_5'] = df['pct_chg'].rolling(window=5).std()
    
    # 3. MACD 及其信号线 (手动计算，避免依赖 talib)
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # 4. RSI (相对强弱指数)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    # 5. 动量 (Momentum): 现在的价格 / 10天前的价格
    df['mom_10'] = df['close'] / df['close'].shift(10) - 1

    # --- 标签构造 (Labeling) ---
    # 预测目标：明天的收益率 (T+1)
    df['target'] = df['pct_chg'].shift(-1)
    
    # 清洗空值 (因 rolling 和 shift 产生的 NaN)
    df = df.dropna()
    
    # 选取因子列
    features = ['pct_chg', 'vol_5', 'macd', 'signal', 'rsi', 'mom_10']
    
    print(f"数据清洗完成，有效样本数: {len(df)}")
    return df, features

# ==========================================
# 3. 数据集构建 (Sliding Window)
# ==========================================
def create_dataset(df, features, seq_len):
    # 标准化 (Z-Score)
    scaler = StandardScaler()
    data_x = scaler.fit_transform(df[features].values)
    data_y = df['target'].values
    
    X, y = [], []
    for i in range(len(data_x) - seq_len):
        X.append(data_x[i : i + seq_len])
        y.append(data_y[i + seq_len]) # 预测序列结束后的那一天
        
    return np.array(X), np.array(y), scaler

# ==========================================
# 4. Transformer 模型定义
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class QuantTransformer(nn.Module):
    def __init__(self, config):
        super(QuantTransformer, self).__init__()
        self.embedding = nn.Linear(config.feature_dim, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, 
            nhead=config.nhead, 
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.decoder = nn.Linear(config.d_model, 1) # 回归预测收益率

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        output = self.transformer(x)
        # 取最后一个时间步的输出
        return self.decoder(output[:, -1, :]).squeeze()

# ==========================================
# 5. 主程序 (Main)
# ==========================================
if __name__ == "__main__":
    cfg = Config()
    
    # 检查 Token 是否已填写
    if '粘贴' in cfg.tushare_token:
        print("\n!!! 错误: 请先在代码顶部的 Config 类中填入你的 Tushare Token !!!\n")
        exit()

    # 1. 获取数据
    df, feature_names = get_data(cfg)
    if df is None: exit()

    # 2. 准备数据集
    X, y, scaler = create_dataset(df, feature_names, cfg.seq_len)
    
    # 划分训练集 (80%) 和 测试集 (20%) - 严禁 Shuffle
    train_size = int(len(X) * 0.8)
    X_train, X_test = torch.FloatTensor(X[:train_size]), torch.FloatTensor(X[train_size:])
    y_train, y_test = torch.FloatTensor(y[:train_size]), torch.FloatTensor(y[train_size:])
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    
    # 3. 初始化模型
    model = QuantTransformer(cfg).to(cfg.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # 4. 训练循环
    print("\n--- 开始训练 Transformer 模型 ---")
    loss_history = []
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(cfg.device), batch_y.to(cfg.device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{cfg.epochs}] Loss: {avg_loss:.6f}")

    # 5. 预测与回测分析
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(cfg.device)).cpu().numpy()
        actuals = y_test.numpy()

    # IC分析 (Information Coefficient)
    ic = np.corrcoef(preds, actuals)[0, 1]
    
    # 简单策略回测: 预测收益率 > 0 则持有，否则空仓
    # 注意: 这里的 preds 是对 "明天" 的预测，所以信号作用于 "明天" 的真实收益 (actuals)
    strategy_returns = np.sign(preds) * actuals
    cumulative_strategy = np.cumsum(strategy_returns)
    cumulative_benchmark = np.cumsum(actuals)
    
    print(f"\n--- 回测结果 ---")
    print(f"测试集 IC: {ic:.4f}")
    print(f"策略累计收益: {cumulative_strategy[-1]*100:.2f}%")
    print(f"基准累计收益: {cumulative_benchmark[-1]*100:.2f}%")

    # 6. 绘图 
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_benchmark, label='Benchmark (CSI 300)', linestyle='--', alpha=0.6)
    plt.plot(cumulative_strategy, label='Transformer Strategy', color='red', linewidth=2)
    plt.title(f'Transformer Quant Strategy Backtest (Test Set) | IC: {ic:.3f}')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show() # 运行后会弹出一个窗口显示图表