## INTRODUCTION

This is an open source AI Chat bot, written in python with Voice command and Computer vision capabilities. Some of the
AI's computer vision capabilities, includes capturing the computer desktop, webcam and copying the clipboard of the chat. The AI will then describe what it sees and gather the most relevant information from the photo and provide descriptive context. <br/>

However, i did found a possible bug within the webcam photos with continuous capture, as it doesn't overwrite the previous image
and will still provide context of the previous image. I'm still working on fixing the issue, however, i found that it could be a limitation of the LLM Model that i use. Please feel free to review and fork the repo and request a PR if you ever found a solution to it. <br/>

If you found any other issues or bug report, please submit it in issue link. 
<br/><br/>

## INSTALL 

RealTime TTS <br/>
pip install realtimetts[all] <br/><br/>

Piper TTS and <br/>
issues on python 3.12.x on Ubuntu 24.4 <br/>
https://github.com/rhasspy/piper/issues/509 <br/>
https://github.com/rhasspy/piper/issues/384 <br/>
https://github.com/rhasspy/piper/issues/395 <br/><br/>

pip install piper-phonemize-cross <br/>
pip install piper-tts --no-deps <br/><br/>

Issue on Piper Engine "Unexpected WAV Properties: Channels=1, Rate=22050, Width=2" #250 <br/>
https://github.com/KoljaB/RealtimeTTS/issues/250 <br/>
https://github.com/KoljaB/RealtimeTTS/pull/244 <br/>
<br/>
Fix <br/>
https://github.com/KoljaB/RealtimeTTS/issues/250#issuecomment-2588411234 <br/>

ffmpeg <br/>
pip install ffmpeg <br/>
pip install python-ffmpeg <br/>
sudo apt install ffmpeg <br/>
<br/>

CREATE API KEYS FOR THESE SERVICES AND SET IT IN .env FILE UNDER ROOT <br/>

GROQ API KEY <br/>
GROQ_API_KEY="KEYS HERE" <br/><br/>

OPENAI API KEY <br/>
OPENAI_API_KEY="" <br/><br/>

## INSTRUCTIONS

1. Create a Python Environment <br/>
2. Create Environment file for your environment variables in the root folder named ".env" without the quotes. <br/> 
The main services used in the Chat AI Bot is from Groq, you can just disregard the Google Gen AI and Open AI.
<br/><br/>

**GROQ API KEY** <br/>
GROQ_API_KEY="keys-here" <br/><br/>

**GOOGLE GEN AI API KEY** <br/>
GOOGLE_GENAI_API_KEY="keys-here" <br/><br/>

**OPENAI API KEY** <br/>
OPENAI_API_KEY="keys-here" <br/><br/>

3. Install the necessary dependencies in requirements.txt <br/>
4. Adjust the variables for the Chat AI Bot name and the prompt according to your needs, however the prompt i made is more than enough. <br/> 
5. Adjust the LLM model and TTS Engine accordingly. For the TTS, i currently used the RealtimeTTS KokoroEngine for the out
of the box lowend GPU compatibility.  <br/>
6. Enjoy! 
<br/><br/>

## SCREENSHOTS
