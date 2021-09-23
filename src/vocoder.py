import torch, os
import numpy as np
from synthesis import build_model
from synthesis import wavegen
import soundfile as sf

WAVENET_PATH = "../pretrain_model/wavenet.pth"
SPECT_PATH = "../conversion"


def main():
    # model load
    model = build_model().to(torch.device("cuda"))
    model_checkpoint = torch.load(WAVENET_PATH)
    model.load_state_dict(model_checkpoint['state_dict'])

    _, _, file_list = next(os.walk(SPECT_PATH))

    for file in file_list:
        melsp = np.load(os.path.join(SPECT_PATH, file))
        print(file[:-4], melsp.shape)
        print(melsp[5][0:3])
        waveform = wavegen(model, c=melsp)
        sf.write(file[:-4] + ".wav", waveform, 16000)


if __name__ == '__main__':
    main()