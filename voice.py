import simpleaudio as sa
import speech_recognition as sr

test_path = "./emo/name_Required.wav"

def play_sound():
    wave_obj = sa.WaveObject.from_wave_file(test_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def speech2text(filename):
    r = sr.Recognizer()
    text = ""
    with sr.AudioFile(filename) as source:
        # listen for the data (load audio to memory)
        audio_data = r.listen(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data=audio_data)
    return text


# play_sound()
# print(speech2text(test_path))