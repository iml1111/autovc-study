import torch
import pickle
from synthesis import build_model
from synthesis import wavegen
import soundfile as sf

spect_vc = pickle.load(open('results.pkl', 'rb'))
device = torch.device("cuda")
model = build_model().to(device)
checkpoint = torch.load("checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])

print(len(spect_vc))
for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(c.shape)
    print(name)
    print(c[5][0:3])
    waveform = wavegen(model, c=c)   
    sf.write(name + '.wav', waveform, 16000)