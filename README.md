# autovc-study
autovc-study

# Get Started
```python
$ python make_spectrogram.py
$ python make_speaker_embedding.py
$ python conversion.py --src <식별자> --trg <식별자> --src_spect <spect 파일 이름>
$ python vocoder.py
```

111: 신희재
112: 견자희
2XX: 제공된 샘플 데이터 (외국인)
9XX: 학습이 완료된 샘플 데이터 (외국인)

- 111 -> 112 : 실패
초반 조영재는 들리는 것 같은데 거의 망한듯.

- 111 -> 111 : 대실패
복원은 개뿔이고, 그냥 여자 목소리로 바뀌었다? 왜?

- 112 -> 112 : 대실패
뜬금없은 다른 여자 목소리가 들림
CE, SE에서 완전한 제로샷은 모두 실패.

- 225 -> 226 : 성공 
굉장히 자연스러움 (작은 샘플 데이터로 만든 임베딩임에도 잘 작동함)

- 226 -> 112 : 거의 실패
초반 영단어가 몇개 들리지만 소리가 뚝뚝 끊김 (스피커 인코더에도 학습이 중요한듯)

- 925 -> 928 : 성공

- 112 -> 925 : 거의 실패
악센트는 복원을 한거 같은데 전혀 한국어로 들리지 않음

- 111 -> 928 : 쪼금 성공?
발음은 영어발음인데 나름 바뀌었다?

- 928 -> 111 : 실패
내 목소리를 조금도 흉내내지 못함. 소리 자체는 어느정도 유지한듯
(스피커 인코더도 학습이 필요???)