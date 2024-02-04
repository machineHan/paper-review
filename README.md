# TOWARDS PRACTICAL AND EFFICIENT IMAGE-TO-SPEECH CAPTIONING WITH VISION-LANGUAGE PRE-TRAINING AND MULTI-MODAL TOKENS

## Abstract

논문 제목 그대로 실용적이며 효율적인 이미지-스피치 캡션 모델을 만드는 방법에 대해 소개한다.  
먼저 vision-language model의 knowledge와 image comprehension을 im2sp로 가져온다. 그리고 im2sp의 출력을 discretized speech unit으로 설정한다.  

Speech unit은 언어적 정보가 담겨있다. 이런 speech unit의 특성이 Im2sp의 spoken language modeling과 비전언어 모델의 language modeling capability을 통합하게 한다.  

Speech unit과 비슷하게 image도 vector quantization을 통해image unit으로 변환시킨다.  

Raw image data 대신 Image unit의 사용으로 인해  이미지 데이터를 저장하는데 필요한 데이터 용량이 크게 줄어들었다.  

<br>


## 1. Intro

이미지에 대한 speech description을 합성하는 것은 사람들의 일상생활을 개선하는데 도움이 된다.  

im2sp는 입력 이미지에 대한 문장을 예측하는 image captioning와 연관되어 있다. 
하지만 im2sp는 아직까지 image captioning보단 안 좋다.  
왜냐하면 text-based image captioning(=>supervised하기 쉬움)에 비해, 이미지 이해에 대한 speech regression의 weak supervised이기 때문이다.  

음성 데이터에는 언어적 정보 말고도 이미지에 관계없는 다양한 정보(화자 특성, 말하는 속도, 잡음)들이 포함되어 있기 때문에,  이미지의 실제 정보와 유사하게 speech feature를 만들도록 최적화하는 regression criteria로 모델을 훈련하는 것이 어렵다.  

즉, 훈련에 방해가 되는 데이터가 음성 데이터에 섞여 있어서, 이를 가지고 학습하면 이미지 컨텐츠를 확실하게 익힐 수 없다.  

수 많은 speech task에서 discretized speech unit이 큰 가능성을 보여줬다.  

Speech unit은 self-supervised speech model에서 나온 speech features를 양자화 해서 얻을 수 있다. 이렇게 생성된 Speech unit은 언어적 특성을 담고 있기에, pseudo-text로 다뤄질 수 있다. 이런 특성을 활용해, regression criterion이 아닌 discrete supervision으로 모델을 훈련 시킬 수 있다.  
아까 speech data의 설명중에 관계없는 정보가 많아 regression criterion으로 훈련하기 어렵다고 서술한 바가 있다. 그래서 speech unit이 더욱 곽광받고 있다.  

하지만 이렇게 훈련 시켜도 image captioning에는 훨씬 못 미쳐, Real-world에 활용이 부적절하다.  

이미지에 맞는 음성데이터를 갖추는 것이 어려우므로,  모델이 이미지에 대해 어떻게 이해할지, 어떻게 이미지를 speech description으로 변환할지 가르치는게 어렵다.  

이렇게 데이터 자체가 적더라도 image-speech를 연관시키는 방식은 연구해야한다.  

<br>

이 논문은 end-to-end im2sp model 성능 향상에 목적을 둔다.  
end-to-end란 speech task에서 특별이 사용된다. speech data에 대한 정보 추출 모듈이 없이 바로 raw data가 모델에 바로 들어가는 것이다. 
보통은 speech data에 대한 feature를 추출하는 작업이 선행됨.  

우리는 pre-trained image-text model의 image understanding과 language generation이 im2sp모델에 전달 될 수 있는지를 먼저 조사한다. 
그 후, 비록 image-text model이 음성데이터로 훈련되지 않았지만, 이미 학습된 특성이 im2sp model의 성능을 향상시켰다.  

어떻게 im2sp model의 효율성을 강화시켰는지 설명하겠다.  

그리고 speech unit과 비슷하게 raw image 대신 image unit을 사용하겠다.  
이미지를 토큰으로 만들고 토큰에 Vector Quantization를 해서 image unit을 만들었다. 
이렇게 토큰을 만들어 사용하면 im2sp model이 언어번역하는 NLP task와 같아진다. 우리 시스템은 결국 image unit을 받아 speech unit을 뱉는 language translation과 비슷하다.  

입력 정보가 모두 토큰이기에 입/출력값이 모두 이산화 되어 모델훈련 + 저장공간 모두 효율적이다. 
효율적인데 image-text의 학습된 정보를 사용해서 im2sp의 성능도 좋아짐!  


이 논문은 pre-trained image-text model를 im2sp에 사용하는 첫 논문이다.  
im2sp model에 pre-trained vision-language model의 image encoder과 text decoder를 가져온다.  
image token을 speech token으로 변역하는 NLP-like processing of multi-modality이다. 이는 저장공간을 많이 줄이는 방식이다.  
여러가지 성능 평가를 통해 이 im2sp model이 주어진 image에 걸맞는 좋은 speech를 만드는 것을 보여주겠다.  

<br>

## 2. METHOD

우리의 목표는 입력 이미지에 알맞는 speech description을 만들어 내는 것이다. Input image x는 (H,W,C) 차원 output speech caption은 (T) 이다.  
T = length of the waveform  

주목표는 input image x를 output speech description으로 바꾸는 것이다. 성능 향상을 위해 large scale pre-trained vision-language model의 Knowledge를 활용한다. Multimodal token(image token, speech token)을 사용하여 효울성도 높인다.  

<br>

### 2.1 Speech Unit Extraction

이전 im2sp 모델들은 discreted acoustic unit을 다시 복구시킨후에 prediction을 만든다.  
우리는 speech unit을 사용해 모델 성능을 높인다. Speech unit에서 직접 추출하는 것으로 인해, speech unit의 linguistic modeling에 더 집중 할 수 있기 때문이다. speech에 다른 요소(모델 성능향상에 의미 없는 요소)는 억제하면서.  

> speech unit이 언어적인 특성을 많이 담고 있으므로, 계속 언어적인 특징을 계속 강조함.  그리고 이런 특성으로 인해 VLMs의 Decoder 부분을 그냥 바로 사용할 수 있는 것이다.


다른 이전 실험은 Mel-spectrogram에서 생성된 discrete acoustic unit을 사용했지만, 우린 최근 모델 HuBERT에서 발견한 speech unit을 사용한다.  

Speech의 feature extraction과정이 Mel-spectrogram 에서 HuBERT로 바뀌어서 복잡한 과정들이 모두 다 사라짐  


우리는 speech unit-based vocoder를 이용해 speech unit을 바로 waveform으로 변환한다.  

다음은 전반적인 모델 과정의 요약본
1. Raw image > image unit : using ViT-VQGAN(vector Quantization) using K means clustering
2. image unit > speech unit : im2sp system tranfered from pre-trained vision-language model
3. speech unit > waveform : speech unit-based vocoder


우리는 end-to-end Model로 speech feature extract를 pre-trained HuBERT에서 진행할 것이다. 그리고 Discretized unit을 얻기 위해 clustering을 할 것 이다. 여기다 연속적인 반복 유닛을 지우면, 최종적으로 im2sp model에서 사용할 speech unit을 얻을 수 있다.  

요약하자면 im2sp model을 훈련할 때 사용할 speech unit은 다음과 같이 만들어 진다.

    raw data > HuBERT > K means clustering > remove repetition > speech unit 

<br>


### 2.2 Image-to-Speech with Vision-Language Pre-trainging

Im2sp model은 Image encoder, speech decoder 이렇게 2개로 구성되어있다.  

Image encoder는 Vit로 만들어짐. Image encdoer에게 이미지가 주어지면 결과값이 downsampling됨. Image encoder에서 나온 visual feature를 flattened하고 이를 speech decoder가 사용한다.  

Visual feature + BOS + speech unit을 speech decoder에게 입력하여, speech unit을 생성한다.  


Vision-language pre-training에 영감을 받아, 우리 im2sp model에 vision-language model capability를 가져오도록 실험했다.

Imaeg-speech data pair가 적은 상황에서 이 translation 는 확실히 도움이 됐다.  

특히 im2sp model의 image encoder, speech decoder는 GiT model로 부터 초기화 되었다. GiT model이란  이미지의 생성 텍스트로 훈련됐다. GiT은 이미지를 어떻게 이해해야하는지, 어떻게 언어로 설명해야 할지를 학습했다.  

Im2sp model의 speech decoder가 GiT의 text decoder로부터 초기화 됐다는 사실을 다시 한 번 말한다. 우리가 사용할 speech unit은 주로 언어적인 정보를 가지고 있기에, vision-language model Text decoder의  language modeling ability를 im2sp의 speech decoder에 옮길 수 있었다.  

<br>

### 2.3 Efficient Image-to-Speech Captioning with Image Units

멀티모달 환경에서 특히 image-audio 상황에서는, text-only보다 더 많은 데이터 용량과 memory cost 가 필요하다. 이것이 왜 NLP보다 speech processing 어려운지를 나타낸다(data set의 불충분 + 엄청난 용량). 요즘은 Image feature에 vector Quantization을 적용해 content를 유지하며 discrete representation을 압축하는 방식이 나왔다.  

이 기술을 사용해, 우린 quantized image representation, image unit, 을 사용하겠다.  

이러면 우리 모델은 discrete image token을 입력으로 받고 discrete speech token은 출력으로 생성한다. NLP task랑 모양이 좀 비슷하다.  

Image unit을 만드는 pre-trained ViT-VQGAN을 사용한다. 하지만 앞에서 말한데로 vision-language model의 knowledge를 사용할 것이다. vision-language model을 훈련하고 학습된 knowledge를 im2sp로 전달한다. 입력데이터를 변경(raw image > image unit)하기 위해 SEiT, Stem-Adpator를 활용한다.  

시스템에서 모두 discrete data를 사용하기에 필요한 데이터의 양이 크게 줄어 든다.  

<br>

## summary

VLM에서 가저온 text decoder를 그대로 사용 가능했던 이유는 speech unit이 언어적인 정보를 특별나게 가지고 있기 때문이다. 그래서 아무 변환없이 사용해도 좋은 성능을 볼 수 있었다.  
게다가 pre-trained Git model을 사용해서 language capability역시 받을 수 있어 더 좋은 결과를 냈다.  

이 논문에서 핵심은 image/speech unit을 통해 NLP-like process를 해, speech process 에서 생기는 큰 문제점들을 우회했다는 점이다.

