import wave
import struct
import math

with wave.open("sample.wav", "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(44100)
    for i in range(44100): # 1 second
        val = int(32767 * math.sin(2 * math.pi * 440 * i / 44100))
        f.writeframes(struct.pack('h', val))
print("sample.wav created.")
