import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread

def is_chinese_or_english(s):
    chinese_count = 0
    for c in s:
        if 0x4E00 <= ord(c) <= 0x9FFF or 0x3400 <= ord(c) <= 0x4DBF:
            chinese_count += 1
            break
    return (chinese_count > 0)

# 加载模型和分词器
model_name = "./weights/DeepSeek-R1-Distill-Qwen-14B"  # 替换为实际路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # 使用 BF16 节省显存
    device_map="auto",           # 自动分配 GPU/CPU
    low_cpu_mem_usage=True
).eval()

auth_keys = ["fill your keys here!"]

with st.sidebar:
    #st.title('Deepseek R1 Chatbot Demo')
    fast_api_key = st.text_input("FASTAPI Key", key="chatbot_api_key", type="password")
    
    st.subheader('Models and parameters')
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.6, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.5, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=100, max_value=2048, value=1024, step=4)

    st.markdown('📖 Learn how to locally deploy deepseek in this [blog](https://blog.liuyc.uk/2025/02/10/deepseek-local-deploy/)!')


st.title("Deepseek Distill Qwen 14B Chatbot")
st.caption("🚀 A Streamlit chatbot powered by Liu Yanchao")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "您好，有什么可以帮您？(How can I help you?)"}]

for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message(msg["role"],avatar='./bot-icon.png').write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not fast_api_key:
        st.info("Please add your FASTAPI key to continue.")
        st.stop()
    elif fast_api_key not in auth_keys:
        st.info("Your FASTAPI key is invalid.")
        st.stop()
    else:
        # 输出用户输入的对话prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant",avatar='./bot-icon.png'):
            with st.spinner("Thinking..."):
                # 格式化对话
                #formatted_dialogue = "\n".join([f"{turn['role']}: {turn['content']}" for turn in st.session_state.messages[-2:]])
                # 提示词工程
                if is_chinese_or_english(prompt):
                    prompt = f"你的任务是始终以“<think>\n”开始你的回答，然后接着给出你的理由。接下来是输入：[{prompt}]"
                else:
                    prompt = f'Your task is to always start your response with "<think>\n" followed by your reasoning. Here is the input: [{prompt}]'
                # 编码输入
                inputs = tokenizer(prompt, return_tensors="pt",
                                    truncation=True, max_length=4096).to(model.device)
                placeholder = st.empty()
                
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, decode_kwargs={"errors": "ignore"})
                thread = Thread(target=model.generate, 
                                kwargs={**inputs, "streamer": streamer, 
                                                               "top_p": top_p, 
                                                               "max_new_tokens": max_length,
                                                               "temperature": temperature, 
                                                               "pad_token_id":tokenizer.eos_token_id})
                thread.start()

                
                full_response = '> '
                is_thinking = True
                for text in streamer:
                    if '<think>' in text:
                        is_thinking = True
                        full_response += "> "
                        placeholder.markdown(full_response)
                        continue
                    #print(text)
                    elif '</think>' in text:
                        is_thinking = False
                        #print("Think")
                        full_response += "\n"
                        placeholder.markdown(full_response)
                        continue
                    elif '<｜end▁of▁sentence｜>' in text:
                        #print("EOS")
                        full_response += "!\n"
                        break
                    
                    if is_thinking:
                        text = text.replace('\n', '\n> ')
                    
                    
                    full_response += text
                    placeholder.markdown(full_response)

                placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# nohup streamlit run streamlit_app.py --server.port xxx > streamlit.log 2>&1 &
