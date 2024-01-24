import torchaudio
import torch.nn.functional as F
waveform, sr = torchaudio.load('Nzq88NnDkEk_29.wav')

        # Cut Spec
specgram = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    win_length=int(float(sr) / 16000 * 400),
    n_fft=int(float(sr) / 16000 * 400)
)(waveform).unsqueeze(0)

def cutSpecToPieces(spec, stride=32):
    # Split the audio waveform by second
    total = -(-spec.size(-1) // stride)
    specs = []
    for i in range(total):
        specs.append(spec[:, :, :, i * stride:(i + 1) * stride])

    # Pad the last piece
    lastPieceLength = specs[-1].size(-1)
    if lastPieceLength < stride:
        padRight = stride - lastPieceLength
        specs[-1] = F.pad(specs[-1], (0, padRight))

    return specs

specgrams = cutSpecToPieces(specgram)


print(1)


