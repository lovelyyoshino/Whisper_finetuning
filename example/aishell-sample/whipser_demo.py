import whisper

def transcribe_audio(file_path, model_path):
    # 加载模型，指定模型文件路径
    model = whisper.load_model(model_path, device="cpu")

    # 读取音频文件并进行识别
    result = model.transcribe(file_path)

    # 输出识别结果
    print("训练识别的文本:", result['text'])

    # 加载模型，指定模型文件路径
    model = whisper.load_model("tiny", device="cpu")

    # 读取音频文件并进行识别
    result = model.transcribe(file_path)

    # 输出识别结果
    print("未训练识别的文本:", result['text'])


# 指定你的音频文件路径
audio_file_path = "C://Users//pony//Desktop//WhisperMultitaskFinetuning//example//aishell-sample//data//wav//BAC009S0150W0001.wav"
# 指定模型文件路径
model_file_path = "C://Users//pony//Desktop//WhisperMultitaskFinetuning//example//aishell-sample//model//tiny.pt"

# 调用函数
transcribe_audio(audio_file_path, model_file_path)
