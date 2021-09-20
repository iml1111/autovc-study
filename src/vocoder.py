import click
import torch
import numpy as np
from synthesis import build_model
from synthesis import wavegen
import soundfile as sf

WAVENET_PATH = "../pretrain_model/wavenet.pth"


@click.command()
@click.option("--spect")
def main(spect):
	device = 0 if torch.cuda.is_available() else -1
	model = build_model()
	if device >= 0:
		model.cuda(device)
	model_checkpoint = torch.load(WAVENET_PATH)
	model.load_state_dict(model_checkpoint['state_dict'])

	melsp = np.load(spect)
	waveform = wavegen(model, c=melsp)
	sf.write(spect[:-4] + ".wav", waveform, 160000)


if __name__ == '__main__':
	main()