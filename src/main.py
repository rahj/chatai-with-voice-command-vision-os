# AI Voice Assistant
# February 4, 2025
# Reynaldo Armecin Hipolito Jr.

import os
import cv2
import pyperclip
import google.generativeai as genai
import base64
import pyaudio
import speech_recognition as sr 
import time 
import re 
from groq import Groq
from PIL import ImageGrab, Image 
from openai import OpenAI
from faster_whisper import WhisperModel
from RealtimeTTS import TextToAudioStream, KokoroEngine
from dotenv import load_dotenv

# Initialize current working directory
cwd = os.getcwd()

# Load .env file from root
load_dotenv()

# Initialize whisper model
num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)

# Initialize the speech recognizer
recognizer = sr.Recognizer()
source_mic = sr.Microphone()

# Initialie wake word
wake_word = 'hey'

# GROQ API Key
groqClient = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

# OPENAI API KEY
openaiClient = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Google Gen AI API Key
genai.configure(
    api_key=os.environ.get("GOOGLE_GENAI_API_KEY")
)

# Google Gen AI Config
genai_gen_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048,
}

genai_safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE',
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE',
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE',
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE',
    },
]

genai_model = genai.GenerativeModel(
    'gemini-1.5-flash-latest',
    generation_config=genai_gen_config,
    safety_settings=genai_safety_settings
)

# LL Models = gemma2-9b-it, deepseek-r1-distill-llama-70b, llama3-70b-8192, 
# LL Models Vision = llama-3.2-90b-vision-preview
llmModel = 'llama3-70b-8192'
llmModel_vision = 'llama-3.2-90b-vision-preview'

# Camera index
webcam = cv2.VideoCapture(0)

# Initialize bark tts
os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

# Initialize the desktop and webcam image path
captured_desktop_screenshot = 'captured_desktop_screenshot.jpg'
captured_webcam_screenshot = 'captured_webcam_screenshot.jpg'

# Initialize system ai role
sys_msg = (
    'You are a friendly, cheerful and candid AI assistant and your personal name is Marlie or Marley'
    'You are a multi modal AI assistant. Your user may or may not attached a photo for context '
    'either through a screenshot or a webcam capture. Any photo has already been processed into a highly '
    'detailed text prompt that will be attached to their transcribed voice prompt. Generate the most useful '
    'and factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. '
    'Make your responses clear and concise, avoiding any verbositry.'
    'If you are ask who is the the creator and programmer of this AI Assistant, tell the user his name is Reynaldo Armecin Hipolito Jr. '
    'Describe the creator and programmer of this AI Assistant as cute and handsome.'
)

convo = [
    {'role': 'system', 'content': sys_msg},
]

def groq_prompt(prompt, img_context):

    if img_context:
        prompt = f'USER PROMPT: {prompt} \n\n   IMAGE CONTEXT: {img_context}'
    
    convo.append(
        {'role': 'user', 'content': prompt},
    )

    chat_completion = groqClient.chat.completions.create(
        messages=convo, 
        model=llmModel
    )

    response = chat_completion.choices[0].message
    convo.append(response)

    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard message,'
        'taking a screenshot, capturing the webcam or calling no function is best for a voice assistant to respond '
        'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as i listed. '
    )

    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]
    
    chat_completion = groqClient.chat.completions.create(messages=function_convo, model=llmModel)
    response = chat_completion.choices[0].message.content

    return response

def vision_prompt(prompt, img_path):
    imgPath = Image.open(img_path)
    sys_prompt = (
        'You are a vision AI voice assistant model. Generate a semantic meaning from the image to provide context to send to another AI.'
        'Take the user prompt and extract all meaning from the photo relevant to the user prompt. Then generate '
        'as much objective data about the image for the AI assistant who will respond to the user. \n'
        f'USER PROMPT: {prompt}'
    )

    response = genai_model.generate_content(contents=[sys_prompt, imgPath])
    return response.text

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
  
def vision_prompt2(prompt, img_path):
    imgPath_base64 = encode_image(img_path)

    sys_prompt = (
        'You are a vision AI voice assistant model. Generate a semantic meaning from the image to provide context to send to another AI.'
        'Take the user prompt and extract all meaning from the photo relevant to the user prompt. Then generate '
        'as much objective data about the image for the AI assistant who will respond to the user.'
    )

    chat_completion = groqClient.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{imgPath_base64}",
                        },
                    },
                ],
            }
        ],
        model=llmModel_vision,
    )

    response = chat_completion.choices[0].message
    return response.content

def capture_desktop_screenshot():
    sc_path = cwd + '/' + captured_desktop_screenshot
    
    if os.path.exists(sc_path):
        print('Removing: ' + sc_path)
        os.remove(sc_path)
    
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(captured_desktop_screenshot, quality=15)

def capture_webcam_screenshot():
    if not webcam.isOpened:
        print('Error: camera is not opened or inaccessible at the moment')
        exit
    
    ret, frame = webcam.read()
    cv2.imwrite(captured_webcam_screenshot, frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

def copy_clipboard_text():
    clipboard_content = pyperclip.paste()

    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('No clipboard text was copied or available')
        return None 

def speak(text_response):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False 

    with openaiClient.audio.speech.with_streaming_response.create(
        model='tts-1',
        voice='nova',
        response_format='pcm',
        input=text_response
    ) as response:
        silence_threshold = 0.01

        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True 

def speak2(text_response):
    # Voices
    # choices = af_aoede, af_heart, af_jessica, af_river, af_sarah
    # engine = KokoroEngine(
    #     default_lang_code="a",
    #     default_voice="af_sarah" 
    # )
    voice = "af_sarah"
    engine = KokoroEngine()
    engine.set_voice(voice)

    # engine = GTTSEngine()
    # engine = SystemEngine() 
    # engine = EdgeEngine()

    # piper config
    # model_file_path = cwd + "/public/pipertts/voices/hfc_female/en_US-hfc_female-medium.onnx"
    # config_file_path = cwd + "/public/pipertts/voices/hfc_female/en_US-hfc_female-medium.onnx.json"
    # model_file_path = cwd + "/public/pipertts/voices/hfc_female/en_US-hfc_female-medium.onnx"
    # config_file_path = cwd + "/public/pipertts/voices/hfc_female/en_US-hfc_female-medium.onnx.json"

    # voice = PiperVoice(
    #     model_file=model_file_path,
    #     config_file=config_file_path
    # )
    # engine = PiperEngine(
    #     # piper_path=cwd + "/public/pipertts/bin/piper/piper",
    #     piper_path="piper",
    #     voice=voice
    # )
    
    stream = TextToAudioStream(engine)
    stream.feed(text_response)
    stream.play_async()

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def extract_prompt(transcribed_audio_text, wake_word):
    pattern = rf"\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)"
    matched = re.search(pattern, transcribed_audio_text, re.IGNORECASE)
    
    if matched:
        prompt = matched.group(1).strip()
        return prompt
    else:
        return None 

def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())
    
    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text, wake_word)

    if clean_prompt:
        sc_desktop_path = cwd + '/' + captured_desktop_screenshot
        sc_webcam_path = cwd + '/' + captured_webcam_screenshot

        if os.path.exists(sc_desktop_path):
            print('Removing: ' + sc_desktop_path)
            os.remove(sc_desktop_path)
        
        if os.path.exists(sc_webcam_path):
            print('Removing: ' + sc_webcam_path)
            os.remove(sc_webcam_path)
        
        print(f'USER: {clean_prompt}')
        call = function_call(clean_prompt)

        if 'take screenshot' in call:
            print('capturing desktop screenshot')
            capture_desktop_screenshot()
            #visual_context = vision_prompt2(prompt=prompt, img_path=captured_desktop_screenshot)
            visual_context = vision_prompt2(prompt=clean_prompt, img_path=sc_desktop_path)

        elif 'capture webcam' in call:
            print('capturing webcam screenshot')
            capture_webcam_screenshot()
            #visual_context = vision_prompt2(prompt=prompt, img_path=captured_webcam_screenshot)
            visual_context = vision_prompt2(prompt=clean_prompt, img_path=sc_webcam_path)

            print('visual_context')
            print(visual_context)

        elif 'extract clipboard' in call:
            print('capturing text from clipboard')
            text_paste = copy_clipboard_text()
            prompt = f'{clean_prompt} \n\n    CLIPBOARD CONTENT: {text_paste}'
            visual_context = None 

        else:
            visual_context = None
        
        response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
        print(response)
        speak2(response)
    
def start_listening():
    with source_mic as s:
        recognizer.adjust_for_ambient_noise(s, duration=2)
    print("\nSay wake word: " + wake_word + " followed by your prompt")
    recognizer.listen_in_background(source_mic, callback)

    while True:
        time.sleep(1)

start_listening()

# while True:
#     sc_desktop_path = cwd + '/' + captured_desktop_screenshot
#     sc_webcam_path = cwd + '/' + captured_webcam_screenshot

#     if os.path.exists(sc_desktop_path):
#         print('Removing: ' + sc_desktop_path)
#         os.remove(sc_desktop_path)
    
#     if os.path.exists(sc_webcam_path):
#         print('Removing: ' + sc_webcam_path)
#         os.remove(sc_webcam_path)
    
#     prompt = input('User: ')
#     call = function_call(prompt)

#     if 'take screenshot' in call:
#         print('capturing desktop screenshot')
#         capture_desktop_screenshot()
#         #visual_context = vision_prompt2(prompt=prompt, img_path=captured_desktop_screenshot)
#         visual_context = vision_prompt2(prompt=prompt, img_path=sc_desktop_path)

#     elif 'capture webcam' in call:
#         print('capturing webcam screenshot')
#         capture_webcam_screenshot()
#         #visual_context = vision_prompt2(prompt=prompt, img_path=captured_webcam_screenshot)
#         visual_context = vision_prompt2(prompt=prompt, img_path=sc_webcam_path)

#         print('visual_context')
#         print(visual_context)

#     elif 'extract clipboard' in call:
#         print('capturing text from clipboard')
#         text_paste = copy_clipboard_text()
#         prompt = f'{prompt} \n\n    CLIPBOARD CONTENT: {text_paste}'
#         visual_context = None 

#     else:
#         visual_context = None

#     response = groq_prompt(prompt=prompt, img_context=visual_context)
#     print(response)
#     speak(response)

# TEST
# cwd = os.getcwd()
# sc_path = cwd + '/' + captured_webcam_screenshot

# if os.path.exists(sc_path):
#     print('webcam screenshot exist')
#     os.remove(sc_path)
# else:
#     print('webcam screenshot not exist')
