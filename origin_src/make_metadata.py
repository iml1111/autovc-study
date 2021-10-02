"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

# Speaker Encoder
C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()

"""
model_b 부분을 꺼내서, C(Speack Embedding) 모델로 불러옴
"""
c_checkpoint = torch.load('3000000-BL.ckpt')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val

C.load_state_dict(new_state_dict)
num_uttrs = 10
len_crop = 128

# Directory containing mel-spectrograms
rootDir = './spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # make speaker embedding
    assert len(fileList) >= num_uttrs
    # train data를 랜덤으로 가져오기 위해서 섞은 듯
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    for i in range(num_uttrs):
        # spectrogram 불러오기 = tmp (300~500, n_mels)
        tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
        
        candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
        # choose another utterance if the current one is too short
        # 현재 발화 데이터가 너무 짧을 경우, 다른 발화 데이터를 대체
        while tmp.shape[0] < len_crop:
            idx_alt = np.random.choice(candidates)
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
            candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
        
        # 발화 데이터 내에서 특정 부분을 발췌함(crop)
        left = np.random.randint(0, tmp.shape[0]-len_crop)
        # melsp = (1, len_crop, n_mels)
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
        # emb = (1, 256)
        emb = C(melsp)
        # emb = (256,)
        embs.append(emb.detach().squeeze().cpu().numpy())
    utterances.append(np.mean(embs, axis=0))
    
    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)
    
with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)
"""
speakers 구조: train.pkl
[
    [
        "p225",
        Ndarray(256,),
        "각각 파일 위치", "각각 파일 위치", "각각 파일 위치", "각각 파일 위치", ...
    ],
    ...
]
"""