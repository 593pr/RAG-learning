"""加载 .env 配置"""

import os
from dotenv import load_dotenv

load_dotenv()

QWEATHER_KEY = os.getenv("QWEATHER_KEY")

if not QWEATHER_KEY:
    raise ValueError("❌ 未找到 QWEATHER_KEY，请在 .env 文件中设置你的和风天气 API Key")