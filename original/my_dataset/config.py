import pathlib


class Config:
    # 修正路径配置
    NPY_DIR = pathlib.Path("E:/Jupyter/track/singletarget/duixiaohuibo")
    XLS_DIR = pathlib.Path("E:/Jupyter/track/t_w_r_label")
    DB_PATH = "E:/Jupyter/track/my_dataset/timeseries.db"  # 完整路径

    # 文件扩展名配置
    NPY_EXT = ".npy"
    XLS_EXT = ".xlsx"  # 改为实际使用的.xlsx

    # 数据参数
    SEQ_LENGTH = 5  # 时间序列长度
    OVERLAP = 0  # 重叠步数
    NPY_DIMS = (8, 300)  # NPY数据维度 (除时间维外)

    # 训练参数
    BATCH_SIZE = 32
    NUM_WORKERS = 4