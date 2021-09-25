import os
import re
import tqdm
import soundfile
import librosa

class Converter():
    def __init__(self, in_path, out_path=None, sample_rate=16000, threads=0, cur=0):
        self.in_path = in_path
        self.out_path = out_path if out_path else in_path + "/result"
        self.sample_rate = sample_rate

        # 분할 처리
        targets_len = len(os.listdir(self.in_path))
        if cur == 0:
            self.start = 0
            self.end = targets_len
        else:
            split_n = targets_len // threads
            self.start = split_n * (cur - 1)
            self.end = split_n * cur if cur != threads else targets_len

        # Out_path 검증
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        

    def main(self):
        # 파일 변환 과정
        print(self.start, "~", self.end, "변환중 ...")
        for speaker in tqdm.tqdm(os.listdir(self.in_path)[self.start:self.end]):
            # 숨김파일 및 기본 out_path 경로는 건너띄기
            if speaker[0] == '.' or speaker == "result": continue

            # Out_path에 폴더명 PXXX (P + Speaker)로 생성
            speaker_id = re.compile("\d{4}").search(speaker).group(0)
            px = self.out_path + '/P' + speaker_id
            os.makedirs(px)
            
            # In_path -> Out_path로 Sample_rate 수정 및 파일 이름 변경하여 저장.
            record_path = self.in_path + '/' + speaker
            for record in os.listdir(record_path):
                s_id, script = re.findall("\d{4}", record)
                audio, sr = librosa.load(record_path + '/' + record, sr=self.sample_rate)
                soundfile.write(px + '/P' + s_id + '_' + script + '.wav',
                                audio,
                                self.sample_rate,
                                'PCM_24')


if __name__ == "__main__":
    in_path = "학습 대상 파일들 경로" # ex) /Users/837477/desktop/validation/AI비서/2
    out_path = "전처리 완료 파일들 저장 경로" # ex) /Users/837477/desktop/result -> None 입력시, in_path에 result 폴더에 저장됨.
    sample_rate = 16000
    thread = 4
    cur = 1
    c = Converter(in_path,
                  out_path,
                  sample_rate,
                  thread,
                  cur)
    c.main()
