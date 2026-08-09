
"""
提示：
day 1

def add_contact(name, phone, email):添加联系人
def search(name):搜索联系人，返回详细信息
def list_by_letter(letter):列出所有某字母开头的联系人

嵌套字典：contacts["小明"] = {"phone": "138...", "email": "..."}
遍历用 for name, info in contacts.items():
检查首字母用 name.startswith(letter)

day 2

给通讯录加存盘功能
"""
contacts = {}
import sys
import json
sys.stdout.reconfigure(encoding='utf-8')


try:
    with open("contacts.json","r",encoding="utf-8") as f:
        contacts = json.load(f)
except FileNotFoundError:
    contacts = {}

def save():
    with open("contacts.json","w",encoding="utf-8") as f:
        json.dump(contacts,f,ensure_ascii=False)



def add_contact(name, phone, email):
    contacts[name] = {"phone": phone,"email": email}
    save()


def search(name):
    if name in contacts:
        a = contacts[name]
        return f"电话: {a['phone']},邮箱: {a['email']}"
    else:
        return f"没有找到 {name}"

def list_by_letter(letter):
    result = []
    result = [f"{name} | 电话:{info['phone']} | 邮箱：{info['email']}" for name,info in contacts.items() if name.startswith(letter)]
    return result

add_contact("小明", "13800138000", "xiaoming@qq.com")
add_contact("小红", "13900139000", "xiaohong@qq.com")
add_contact("小刚", "13700137000", "xiaogang@qq.com")

print(search("小明"))        # 应该显示小明的电话和邮箱
print(list_by_letter("小"))  # 应该列出全部三个