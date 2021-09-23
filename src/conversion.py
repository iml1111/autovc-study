import os, click
import pickle
import torch
import numpy as np
from math import ceil
from autovc_model import Generator

AUTOVC_PATH = '../pretrain_model/autovc.ckpt'
SPECT_PATH = "../spect/%s"
EMB_PATH = "../embs"
RESULT_PATH = "../conversion"


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

@click.command()
@click.option("--src")
@click.option("--src_spect")
@click.option("--trg")
def main(src, src_spect, trg):
    device = 0 if torch.cuda.is_available() else -1
    generator = Generator(32, 256, 512, 32).eval()
    if device >= 0:
        generator = generator.cuda(device)
    model_checkpoint = torch.load(AUTOVC_PATH, map_location='cuda:0' if device >= 0 else 'cpu')
    generator.load_state_dict(model_checkpoint['model'])

    uttr_org = np.load(os.path.join(SPECT_PATH % src, src_spect))
    uttr_org, len_pad = pad_seq(uttr_org)
    emb_org = np.load(os.path.join(EMB_PATH, "%s.npy" % src))
    emb_trg = np.load(os.path.join(EMB_PATH, "%s.npy" % trg))

    uttr_org = torch.from_numpy(uttr_org[np.newaxis, :, :])
    emb_org = torch.from_numpy(emb_org[np.newaxis, :])
    emb_trg = torch.from_numpy(emb_trg[np.newaxis, :])

    if device >= 0:
        uttr_org = uttr_org.cuda(device)
        emb_org = emb_org.cuda(device)
        emb_trg = emb_trg.cuda(device)

    print('{}x{}'.format(src, trg))
    print(uttr_org.shape, uttr_org[0][0][:3])
    print(emb_org.shape, emb_org[0][0:3])
    print(emb_trg.shape, emb_trg[0][0:3])

    with torch.no_grad():
        _, x_identic_psnt, _ = generator(uttr_org, emb_org, emb_trg)

    if len_pad == 0:
        uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
    else:
        uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

    print(uttr_trg.shape)
    print(uttr_trg[0][0:3])

    np.save(RESULT_PATH + "/%sx%s" % (src_spect[:-4], trg), uttr_trg)



if __name__ == '__main__':
    main()

