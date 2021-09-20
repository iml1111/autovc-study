"""
Speaker embeddings 생성기
wavs에 존재하는 각 식별자들에 spk emb 데이터 생성
- 현재 존재하는 wavs 내의 파일들의 평균으로 emb 도출.
"""
import os
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import torch
from speaker_encoder_model import SpeakerEncoder


SPEAKER_ENCODER_PATH = "../pretrain_model/speaker_encoder.ckpt"
SPECT_DIR = "../spect"
EMB_DIR = "../embs"
LEN_CROP = 128

device = 0 if torch.cuda.is_available() else -1



# Speaker Encoder 모델 호출 및 state 불러오기
speaker_encoder = SpeakerEncoder(dim_input=80, dim_cell=768, dim_emb=256).eval()
if device >= 0:
	speaker_encoder.cuda(device)
model_checkpoint = torch.load(SPEAKER_ENCODER_PATH, map_location="cpu")
new_state_dict = OrderedDict()
for key, val in model_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
speaker_encoder.load_state_dict(new_state_dict)


dir_name, subdir_list, _ = next(os.walk(SPECT_DIR))
print("mel-spectogram dicrectory:", dir_name)

# p111, p112, p113, ...
for speaker in tqdm(sorted(subdir_list)):
	_, _, file_list = next(os.walk(os.path.join(dir_name, speaker)))

	embs = []
	# p111/1.npy, p111/2.npy, p111/3.npy, ...
	for file in file_list:
		spect = np.load(os.path.join(dir_name, speaker, file))

		if spect.shape[0] < LEN_CROP:
			continue

		# 발화내용에서 일부분(LEN_CROP)을 랜덤 발췌 
		left = np.random.randint(0, spect.shape[0] - LEN_CROP)
		# crop_spect = (1, LEN_CROP, n_mels)
		crop_spect = torch.from_numpy(
			spect[np.newaxis, left:left+LEN_CROP, :]
		)
		if device >= 0:
			crop_spect.cuda(device)

		# emb = (1, 256)
		emb = speaker_encoder(crop_spect)
		embs.append(emb.detach().squeeze().cpu().numpy())

	avg_emb = np.mean(embs, axis=0)
	np.save(os.path.join(EMB_DIR, speaker), avg_emb)



