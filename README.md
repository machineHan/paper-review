# # HMTL: Heterogeneous Modality Transfer Learning for Audio-Visual Sentiment Analysis

## Before read

발화 : 말, speech, 특히 비디오 속에 나오는 말을 칭함.  이 논문은 비디오에서 나오는 말을 utterance(발화)라고 말함  

Source model : text-only modality  

Target model : acoustic-visual modality  

<br>

## Abstract

Multimodal 감정분석은 language based 감정분석에 확장된 접근 방식이다. 여러가지 모달을 적용해 감정 예측을 만들어낸다.  최근,  multimodal 감정분석을 위한 다양한 data funsing method가 제안되고 있다. 대부분의 경우, text data가 가장 중요한 역할을 하고, visual과 acoustic data는 보조 역할을 한다.   

하지만 비디오와 같은 multimedia에서는, 화자의 말이 textual data로 전달 되지 않는다. (멀티미디어에서는 텍스트 정보를 얻기 힘들다, 주로 image-audio 데이터가 메인이 된다) 즉, real-world sentiment analysis에서는 텍스트를 제외한 image-audio sentiment analysis가 중요하다.   하지만 text modality를 포함한 다른 multimodal model에 비해 image-audio sentiment analysis의 성능이 좋지 않다는 것이 실상이다.  

이 논문에서 heterogeneous modality transfer learning를 제안한다. HMTL은 iamge-audio sentiment analysis 성능을 향상시키기 위해, text data의 정보를 source modality로 활용하는 방식이다.  

이 접근에서multimodal representation을 높이기 위해  decoder와 추가적인 학습 기술을 사용하여 source/target modality의 embedding space의 차이를 줄인다.  

최근에 발표된 uni/bi modal method보다 좋은 성능을 기록했다.  

<br>

## Introduction

감정분석은 텍스트를 통해 특정 주체를 향한 저자의 감정을 추출해 내는 방법이다. 택스트에 숨겨진 정보(태도, 생각, 그리고 기분) 을 분석하여 감정을 예측한다. 이 숨겨진 정보는 저자의 감정을 파악하는 데 아주 유용한 정보이다.  딥러닝 기술의 발전으로, 많은 양의 텍스트를 사용한 다양한 감정분석 연구가 진행되고 있다.  

많은 도메인에서, 딥러닝을 사용한 감정 분석이 text polarity classfication에서 좋은 결과를 낳고 있다.
> polarity classification : 텍스트에서 나타나는 작성자의 태도를 분류 : 긍정적, 부정적, 중립

보통 sentiment analysis는  NLP로 구분된다.  

텍스트에서 감정 분석과 다른 표현 방식으로 화자의 감정을 분류하려는 시도가 이루어졌다. 이 시도는 text, audio, image에 대해서 이뤄졌는데 이것이 multimodal sentiment analysis이다.   

다양한 유튜브와 같은 video 매채가 생겨난 뒤로 multimodal 감정분석에 사용할 수 있는 데이터들이 많아졌다.  이런 매채들은 각 speech에 대한 text/iamge/audio를 모두 포함하고 심지어 각 speech마다 label또는 sentiment score가 부여되어 있다.  

Data fusing은 멀티모달 감정분석의 주된 접근방식이다.
> Data fusing : merge data with different shapes, scales, modalities, or information

Data fusing을 통해 합쳐진 정보는 부족한 정보를 보완에 보다 풍부한 정보를 제공한다. 초기에는 data fusing이 다른 모달들을 concatenated 하는 방식이였지만, 딥러닝 아키텍처의 발전으로 다양한 fusing methodologies가 제공되고 있다. Data fusing method는 크게 2개로 분류된다.
- early fusing : 네트워크에 input layer 근처에서 fusing  low-level에서 바로 fusing하고, 합쳐진 데이터를 가지고 추출한다.

- late fusing : 네트워크에 output layer 근처에서 fusing  hidden에서 추출된 각 modal feature들을 fusing한다. 

하지만 최근의 audio-visual 정보를 적게 사용하는 multimodal sentiment analysis는 그들의 text-dependent 특성으로 인해 한계에 부딪힌다.

Raw video에 대한 정확한 text scripts를 얻는 것을 비싸기에,  real world multimodal sentiment analysis는 제한된다.  
대부분의 multimodal sentiment analysis는 훈련 데이터에 텍스트가 포함돼 있다. 

Text가 없는 Audio- sentiment analysis는 낮은 성능을 보인다.  

우리가 제안하는 HMTL은  audio-image sentiment analysis의 성능을 높이기 위해,  훈련에서 text modality를 사용한다. (test시에만 audio-image만 사용)

HMTL은 서로 다른 characteristic과 distribution을 가진 source/target data를 가지고 훈련하는 inter-domain training method이다.  

우린 멀티모달 상황에서도 text-only model이 높은 성능을 보인 다는 점에 주목한다. 그래서 우리는 text-only model의 풍부한 knowledge를 audio-image model에 transfer learning 한다.  즉 text modality가 source가 되고, audio-image modality가 target이 된다.

HMTL은 다음과 같이 진행된다.
Unimodal sentiment analysis model이 텍스트만 가지고 훈련된다. 이 모델은 훈련할 때만 사용한다.  그리고 이 모델은 pre-train 되어있고, 텍스트 모달리티에서 feature를 뽑는데 사용된다. 뽑아진 feature는 audio-image model의 classification performance를 높이는데 사용된다.  

HMTM(heterogeneous modality transfer module)라는 구조를 만들었다. 훈련 상황에서, source model의 knowledge를 target model에서 transfer해주는 역할이다. 이 과정에서 source/target 간의 feature representation의 distribution을 줄이며 transfer를 한다.  

Source model(text-only)에서 얻어진 text representation은 HMTM를 위해  
Soft label로써 제공된다. Target model의 representation은 source model에서 얻어진 representation vector space와 매핑된다.

이 논문에서 강조하는 바는 다음과 같다
1. 우리는 현재 sentiment analysis 연구와는 동떨어지지만, real world task와 가까운 audio-image sentiment analysis의 중요성을 강조한다.    sentiment analysis에서 텍스트의 중요성이 높다 보니, 훈련에서만 text를 사용해서 audio-image의 성능을 높일 것이다.

2. 이번 논문에서는 sentiment analysis란 특정 테스크에 대해서만 다루지만,  source와 target 간의 정보 분포가 불균형한 상황에서도 사용가능하다.

3. Sentiment analysis에 대한 여러가지 데이터 셋에 대해 실험 할 것이다. 기존의 major study와도 비교하겠다.  

결과는 audio unimodal, visual unimodal, and audio-visual bimodal sentiment classification 이렇게 3가지에 대해 보여주겠다.

## Problem defination

일반적인 sentiment analysis 세팅은 textual, visual, acoustic 정보가 담긴 video를 사용하는 것이다. 만약 말하는 것(발화)의 기준이되는 pivot modality를 사용하여 정렬한다면, 동일한 길이의 발화로 구성된 정렬 데이터가 만들어진다.
> 쉽게 말해서, text modality-based align을 하면, text 정보를 많이 담고 있는 데이터를 만들 수 있다.

이런 methodology를 구상한 의도는 audio-image model을 text/image/audio를 모두 포함시켜 훈련시키고 싶었기 때문이다. 그리고 predict 시에는 image-audio만을 포함한 video를 입력받아 해당 데이터에 대한 sentiment analysis를 하는 것이다.  

대부분의 sentiment analysis는 text가 major key고 audio,image는 보조를 하기 때문에 이렇게라도 텍스트를 끼어넣는다.  

word가 가장 중요한 unit이므로, video에서 말을 시작하는 시간(발화시간)에 기준을 둬 각 모달리티를 정렬한다.  

정렬 후, 각 모달리티는 다음과 같은 dimension이 된다.
- text : Tv = (u,m)
- image : Vv = (u,n)
- audio : Av = (u,o)
> v는 video의 발화시간에 따른 index  u는 비디오에서 나오는 발화의 최대 길이  m,n,o는 각 modality의 차원

각 modality는 전처리 차이로 인한 모양의 차이가 존재한다. 하지만 alignment process 떄문에 같은 video length, utterance length를 가진다.  

HMTL은 audio-image sentiment analysis의 성능 향상을 목표로 한다. 훈련시 textual 정보를 heterogeneous transfer learning한다.  

<br>

## Proposed methodology

HMFL은 크게 3단계로 나눠진다. 

1. Textual data(source modality)에서 뽑아온 feature를 통해 source model을 훈련한다. Source model은 GRU라는 층을 포함하고 있다. 여기서 출력된 textual representation은 HMTL에서 중요한 역할로 사용된다. <br> input(text) > target model > output(textual representation)
2. 1번과정에서 pre-trained Source model의 knowledge를 target model에게 transfer한다.  이 과정에서 heterogeneous modality decoder (HM-Decoder)와 heterogeneous modality discriminator (HM-Discriminator)를 사용한다. <br> HM-decoder는 target model에서 얻은 target representation을 입력으로 받고,  source representation을 재구조 타겟으로 사용한다. (Target representation을 source와 유사하게 reconstruction한다.)
   > HM-decoder는 두 embedding space 사이의 correlation을 높인다.
   
   > HM-Discriminator는 target model의 output이 source model의 output과 동일해지도록 학습한다.
4. 1,2번 과정이 끝나면 새로운 acoustic-visual data를 통한 sentiment analysis가 진행된다. 이런 상황에서 target model은 source model의 knowledge를 받아 acoustic-visual 만 존재하는 데이터 셋에서도 좋은 결과를 얻을 수 있다.   

논문에서 제시한 사진 중, HMFL이 적용되기 전과 후를 구분해 text,acoustic,visual data의 embedding space간의 차이를 보여주는 사진이 있다. HMTL를 적용하면 acoustic, visual data가 text data embedding에 인접한 모습이 확실히 보인다. 즉 sentiment analysis에서 핵심 역할을 하는 textual capability를 transfer learning하여 acoustic-visual dataset로도 좋은 classification performance를 보여준다.  

<br>

### A. Network architecture

#### 1) Embedding networks

각 modality에 대한 embedding network가 존재한다. Network는 2개의 양방향 GRU와 dense layer가 포함되어 있다. 통과 순서는 다음과 같다.  

GRU - Dense - GRU  

GRU는 발화 중심의 context-aware feature를 추출해서 representation을 출력한다. GRU는 분류하기 위해 충분한 정보가 없어도 전체 맥락의 정보를 반영함으로 인해 효율적인 분류를 한다.  

dense layer는 특정 차원의 표현으로 표현될 수 있도록 이 정보를 효과적으로 요약한다.  

Dropout이 overfitting을 막기 위해 각 layer에 적용되어 있다.  

각 modality 마다 하나씩 존재한다.(text, image, audio)

<br>

#### 2) Classification network

Classification network는 2개의 dense layer를 사용한다. 하나는 Hidden layer의 activation fuction인 ReLU layer이다. 나머지는 Output layer softmax를 사용한다. Embedding network와 동일하게 dropout이 적용돼 있다. Cross-entropy loss function을 사용한다.  

source, target Model에 하나씩 총 2개 존재한다.

<br>

#### 3) HM-Decoder

여기도 dense layer를 사용한다. ReLU는 hidden layer의 activation function에서 사용되고, Tanh layer는 output layer의 activation function에서 사용된다. HM-Decoder는 target representation이 source representation과 큰 연관이 있도록 훈련을 받는다.  

decoder의 성능이 올라가면 올라갈 수록 둘의 correlation이 커진다. 이로서 target classification performance가 향상된다.  

Decoder는 MSE를 사용한다. 

target embedding representation을 decoder에 넣어서 나온 값과 source embedding representation값을 가지고 MSE를 진행한다. 이 loss 값을 가지고 HM-Decoder와 target embedding network(image, audio)를 훈련한다.

즉, HM-decoder를 지난 target representation이 최대한 source representation과 비슷하게 만들게 끔 HM-decoder를 훈련한다.

<br>

#### 4) HM-Discriminator

Dense layer로 구성되어 있다. ReLU for hidden layer, sigmoid for output layer가 있다. HM-Discriminator는 target/source representation을 가지고 source representation을 1이 되도록, target representation이 0이 되도록 훈련한다. 

Target model output을 HM-Discriminator에 넣어서 0이 될 수 있도록 하는 target embedding network loss가 여기서 등장한다.  

Target embedding network loss
> target embedding network는 target representation이 HM-Discriminator를 거쳤을때 1이 나오게 끔 embedding network를 훈련한다.

HM-Discriminator loss
> source representation가 discriminator를 거쳤을 땐 0에 가깝도록 , target representation은 1에 가깝도록 하는 loss를 가지고 HM-Discriminator를 훈련한다.  비슷하게 target embedding network는

이런 접근법이 source/target의 특징, 분산 차이를 좁혀준다.  

즉, 잘 훈련된 HM-Discriminator는 target과 source를 구별하지 못한다. (= acoustic-visual 만으로 text modality의 특성을 살릴 수 있다.) 이를 통해 target modality로만 성능이 좋은 sentiment analysis가 가능한다.

<br>

#### 5) Attention and data fusion

여기서는 Self-attention을 이용한 uni/bi-modal 환경에서의 fusing method에 대해 다룬다.  

Embedding network로 부터 더 좋은 정보를 얻기 위해 Self-attention mechanism을 채용한다.  
Embedding network를 지나 추출된 representation을 통해 self-attention score를 구한다. 기존의 representation에 self-attention score를 element-wise product를 해서 self-attention representation을 구한다.    

Embedding network에서 나온 representation과 이에 self-attention score를 곱해 만든 self attention-based representation을 concate하여 classification task시 사용한다.  

여기까지는 uni-modal fusing method였다.  

이제 target model의 관점에서 보자. 여기서는 2가지 modality를 이용하여 classification 작업을 해야한다. Self attention을 도입하기 위해선 위의 작업보다 추가적인 처리가 필요하다.   

여기서 사용할 것이 bi-modal attention score이다. Bi-attention score는 다른 Modality로 부터 얻은 representation을 element-wise product, softmax 해야한다.  

두 모달사이의 bi-attention score를 구한뒤 각각의 modal representation에 element-wise product한다. 이러면 각 모달에 맞는 bi attention based representation이 완성된다.  

그리고 uni modal data fusing에서는 raw/self-attention representation을 concate했다면, 여기서는 추가적으로 bi-attention representation를 추가해 concate하여 사용한다.

<br>

### B. Objective function and HMTL algorithm

Target model의 2개의 main objective function이 있다. 

Classification objective function은 단순히 입력에 대한 예측의 Loss이다. (Classification network만을 업데이트 하는 것은 아니다) hyper parameter를 하나 곁들인다.  

Target model embedding network는 먼저 classification task를 할 수 있는 feature representation을 포함하도록(=> feature를 잘 뽑도록) 훈련하고 그리고 source/target이 비슷한 distribution을 갖도록  훈련한다.

Embedding objective function는 위에서 다룬 HM-Discrimination와 HM-Decoder에서 등장한 loss를 사용한다.

Embedding objective function를 통해, target modality만으로 source modality와 비슷하고, 연관 깊은 결과를 낼 수 있다.

종합적으로, training phase에서는 HM-Decoder/Discriminator, target embedding/classification network 총 4개가 훈련된다.  

다음은 어디서 어떤 Loss를 이용해 훈련하는지를 요약한 결과이다.

Network | Training stuff
--- | ---
HM-Decoder | decoder loss
HM-Discriminator | Discriminator loss
Target embedding network | classification, embedding object function
Target classification network | classification object function

전반적인 과정을 요약하자면,
1. Source model을 pre-training한다. 
2. HM-Decoder와 HM-Discriminator를 이용하여 source -> target transfer learning을 한다.
> 특이사항 : 모든 modality에 대한 embedding network를 개별로 존재한다(3개).  Classification network는 source/target 이렇게 2개 존재한다.
3. target model을 사용해 성능 측정

<br>

## Experiments and Discussion

여러가지 baseline model에 대해 unimodal(Textual, visual, acoustic) sentiment analysis, bi-modal sentiment analysis(visual-acoustic)에 대한 성능 비교를 행했다.  

HMTL method는 textual data가 제한된 Modal 환경에서 보다 높은 성능을 보여줬다.  


<br>

## summary

target(image-audio)를 source(text)와 유사한 값을 출력하도록 하는 method이다.  

이 과정에서 GANs architecture를 굉장히 유사하게 모방했다.

HM-Decoder는 source와 target이 최대한 유사하게 표현되게 훈련하는 GANs에서 generator역할을 한다.  

HM-Discriminator는 source는 0(fake), Target은 1(real)로 예상하게끔, 즉 source, target을 구별하게 끔 훈련한다. GANs에 Discriminator에 해당한다.

HM-Decoder/Discriminator는 각자에 맞는 loss에 맞춰 훈련한다.  

Target embedding network, classification network는 2개의 main objective function에 맞춰 훈련한다.




