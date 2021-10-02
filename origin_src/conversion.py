import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

device = 'cuda:0'
# Model 호출 및 autovc 모델에서 state load
G = Generator(32,256,512,32).eval().to(device)
g_checkpoint = torch.load('autovc.ckpt', map_location=device)
G.load_state_dict(g_checkpoint['model'])

print(g_checkpoint.keys())

"""
Metadata
wavs 경로에 있는 사람별 목소리 mel-sprectrograms ???
[
    ["p225", Ndarray(256,), Ndarray(90,80)],
    ["p228", Ndarray(256,), Ndarray(89,80)],
    ["p256", Ndarray(256,), Ndarray(75,80)],
    ["p270", Ndarray(256,), Ndarray(109,80)]
]
"""
metadata = pickle.load(open('metadata.pkl', "rb"))

spect_vc = []

for sbmt_i in metadata:
    x_org = sbmt_i[2]
    """
    90, 89, 75 -> 96
    109 -> 128로 변화 32의 배수에 맞게 padding
    """
    x_org, len_pad = pad_seq(x_org)

    """
    uttr_org: source 발화
    emb_org: source 임베딩
    """
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)
    
    for sbmt_j in metadata:
                 
        """emb_trg: target 임베딩"""  
        emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)
        
        print('{}x{}'.format(sbmt_i[0], sbmt_j[0]))
        print(uttr_org.shape, uttr_org[0][0][0:3])
        print(emb_org.shape, emb_org[0][0:3])
        print(emb_trg.shape, emb_trg[0][0:3])

        with torch.no_grad():
            _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
            
        """
        uttr_trg: 타겟 발화
        padding이 있을 경우, padding 부분 제거
        uttr_trg의 shape는 입력된 uttr_org와 완전히 같음.
        """
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
        
        print(uttr_trg.shape)
        print(uttr_trg[0][0:3])

        spect_vc.append( ('{}x{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg) )
        """
        [
            ['p225xp225', Ndarray(n, 80)],
            ...
        ]
        """
        break
    break
        
        
with open('results.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)