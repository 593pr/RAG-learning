"""
import requests

response = requests.get("https://api.github.com/users/593pr")
print(response.status_code)
data = response.json()
print(data["login"])
print(data["public_repos"])

练习：写一个 GitHub 用户查询器
新建 practice/day3_github.py：

import requests

def get_user_info(username):
    查 GitHub 用户信息
    # 你的代码

def search_repos(keyword):
    搜 GitHub 仓库
    # 你的代码

# 测试
print(get_user_info("593pr"))
print(search_repos("python beginner"))
python
提示：

查用户：https://api.github.com/users/{username}
搜仓库：https://api.github.com/search/repositories?q={keyword}
用 response.status_code 判断成功还是失败
搜仓库返回的是 data["items"]，里面是一个列表

"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests

def get_user_info(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        return f"用户名：{data['login']},仓库数：{data['public_repos']},粉丝数：{data['followers']}"
    if response.status_code == 404:
        return f"该{username}不存在"
    
def search_repos(keyword):
    url = f"https://api.github.com/search/repositories?q={keyword}"
    response = requests.get(url)
    data = response.json()
    result = []
    repos = data["items"][:5]
    for repo in repos :
        result.append(repo['name'])
    return result


print(get_user_info("593pr"))
print(search_repos("python beginner"))

    