import librosa
import numpy as np
import os

class Comparator:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.audio, self.sr = librosa.load(audio_path)
        self.features = self.extract_features()

    def extract_features(self):
        return librosa.feature.mfcc(y=self.audio, sr=self.sr)

    def compare(self, other):
        other_features = other.features
        min_length = min(self.features.shape[1], other_features.shape[1])
        self_features_resized = self.features[:, :min_length]
        other_features_resized = other_features[:, :min_length]
        return np.linalg.norm(self_features_resized - other_features_resized)

    def __str__(self):
        return os.path.basename(self.audio_path)
x = Comparator(r'C:\Users\LE TU QUOC DAT\Documents\GitHub\AudioGuessr\audioguessr\audio\snare.mp3')
y = Comparator(r'C:\Users\LE TU QUOC DAT\Documents\GitHub\AudioGuessr\audioguessr\audio\snare2.mp3')
z = Comparator(r'C:\Users\LE TU QUOC DAT\Documents\GitHub\AudioGuessr\audioguessr\audio\crash.mp3')
print(x.compare(y))
print(x.compare(z))