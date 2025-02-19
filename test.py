import requests

url = "http://localhost:23089/generate"

# JSON 数据
json_data = {
    "prompt": "归并排序算法的流程是什么？",
    "max_length": 512
}

# 自定义请求头
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "faf5b738b295e6ff23cbcd9c34f2079a6a591bd21cd35d5b69f6acf7fd88c7f4"
}

# 发送 POST 请求
response = requests.post(url, json=json_data, headers=headers)

# 打印响应状态码和内容
print("Status Code:", response.status_code)
print("Response Content:", response.text)