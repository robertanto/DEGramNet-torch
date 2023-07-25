import torchaudio.transforms as T

# utils waveform to spectrogram
class waveform_to_Spectrogram:
    def __init__(self, sr=16000, win_sec=0.025, hop_sec=0.010, n_fft=2048):
        win_length = int(win_sec*sr)
        hop_length = int(hop_sec*sr)

        self.transform = T.Spectrogram(
            n_fft,
            win_length,
            hop_length,
            normalized = False
        )

    def __call__(self, waveform):
        spectrogram = self.transform(waveform)
        return spectrogram