"""CLI 天气助手 — 输入城市名，输出实时天气"""

import sys
import logging
import requests
from config import QWEATHER_KEY

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# API 地址
# ============================================================
CITY_API = "https://geoapi.qweather.com/v2/city/lookup"
WEATHER_API = "https://devapi.qweather.com/v7/weather/now"


# ============================================================
# ① 城市名 → 城市 ID
# ============================================================
def get_location_id(city_name):
    """把城市名转成和风天气的 location ID"""
    params = {"location": city_name, "key": QWEATHER_KEY}
    logger.info(f"🔍 查询城市: {city_name}")

    try:
        resp = requests.get(CITY_API, params=params, timeout=10)
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"网络请求失败: {e}")
        sys.exit(1)

    if data.get("code") != "200":
        logger.error(f"城市查询失败: {data.get('code')}")
        sys.exit(1)

#用data.get("code")而不是data["code"]是因为.get() 返回 None 不会崩，data["code"] 会直接抛 KeyError
#!= "200" — 不等于 "200" 就是出问题了。注意 "200" 是字符串，不是数字 200，因为 API 返回的是字符串。

    locations = data.get("location", [])
    if not locations:
        logger.error(f"未找到城市: {city_name}")
        sys.exit(1)

    city = locations[0]
    logger.info(f"✅ 找到: {city['name']} ({city['adm1']}) — ID: {city['id']}")
    return city


# ============================================================
# ② 城市 ID → 天气数据
# ============================================================
def get_weather(location_id):
    """根据 location ID 获取实时天气"""
    params = {"location": location_id, "key": QWEATHER_KEY}
    logger.info("🌤️  获取天气数据...")

    try:
        resp = requests.get(WEATHER_API, params=params, timeout=10)
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"网络请求失败: {e}")
        sys.exit(1)

    if data.get("code") != "200":
        logger.error(f"天气查询失败: {data.get('code')}")
        sys.exit(1)

    return data["now"]


# ============================================================
# ③ 格式化输出
# ============================================================
def format_weather(city, now):
    """把天气数据格式化成好看的输出"""
    return f"""
┌──────────────────────────────────┐
│  📍 {city['name']} ({city['adm1']})
│  🕐 数据更新时间: {now.get('obsTime', '未知')}
├──────────────────────────────────┤
│  🌡️  温度:     {now['temp']}°C
│  🤔 体感温度:   {now['feelsLike']}°C
│  ☁️  天气:     {now['text']}
│  💨 风向:     {now['windDir']}
│  🍃 风力:     {now['windScale']} 级
│  💧 湿度:     {now['humidity']}%
│  🎈 气压:     {now['pressure']} hPa
│  👁️  能见度:   {now['vis']} km
└──────────────────────────────────┘"""


# ============================================================
# ④ 主入口
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("用法: python main.py <城市名>")
        print("示例: python main.py 北京")
        sys.exit(1)

    city_name = sys.argv[1]
    city = get_location_id(city_name)
    now = get_weather(city["id"])
    print(format_weather(city, now))


if __name__ == "__main__":
    main()