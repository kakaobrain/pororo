from pororo import Pororo

asr = Pororo(task='asr', lang='ko')

print(asr('test_audio.wav'))
