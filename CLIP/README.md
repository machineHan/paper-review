# CLIP : Learning Transferable Visual Models From Natural Language Supervision


## 1. Abstract

기존의 비전모델들은 주어진 카테고리(클레스)에 따라  훈련이 된다. 만약 추가적인 비전 컨셉이 필요하다면 추가적인 데이터가 필요하다.  이런 제한된 형태는 generality와 usability의 한계를 만든다. 

<br/>

이미지에 관련된 텍스트로 학습하면 더욱 광범위한 정보를 학습할 수 있다.

<br/>


어떤 캡션이 어떤 이미지와 맞는지 예측하는 간단한 pretraining method은 인터넷에서 모은 이미지 텍스트 데이서 쌍을 from scratch부터 학습하며 SOTA image representation을 배울 수 있다.

<br/>


위의 훈련이 끝나고, 자연어는 학습된 visual comcept의 reference로 사용된다. 또한 Downstream task로 zero-shot transfer가 가능해진다.


<br/>


이 모델은 fully supervised baseline에 비해 특정 데이터셋 훈련(optimizing)이 필요없는데, 좋은 성능이 나온다.

<br/>


## 1.Introduction and Motivation Work

<br/>


Autoregressive, masked language modeling 같은 Task-agnostic 오브젝트는 계산, 모델 용량, 데이터의 규모 측면에서 꾸준히 확장, 향상 시켰다.

<br/>

표준 입출력으로써의 text-to-text의 개발은 task-agnostic 아키텍쳐에서 추가적인 output head나 데이터셋에 대한 추가적인 커스텀 없이 downstream dataset으로 zero-shot transfer를 가능하게 했다. 

<br/>

이제 GPT-3같은 시스템이 많은 테스크에서 경쟁력있는 성능을 내면서, 특정 데이터에 대한 추가 훈련이 거의 필요 없다 <br/>
language model에서의 여러 발전을 통해 NPL dataset보다 무한에 가까운 web-crolled data로 훈련하는게 더 좋아졌다. <br/>
엄청 많은 양의 데이터를 가지고 pre-train하니 overfitting, generalizability 둘다 걱정이 없고, downstream task을 위한 추가적인 Optimization도 필요없다


<br/>


이러한 결과로 높은 퀄리티의 NLP dataset보다도 웹에 긁어온 text set을 가지고 사용 가능한 modern pre-training method 활용한 supervision이 더 좋다는 것을 나타낸다.

<br/>

하지만 text task가 아닌 Image task 부분에서는 아직 이런 특징이 안나타난다. 즉 아직까진 crowd-labeled dataset를 사용하는 것이 일반적이다.

<br/>


오래전부터 현재까지도 text-image에 관한 연구가 많이 진행됐다. 대부분이 text를 통해 image representation을 더욱 잘 학습할 수 있다고 설명했고 이를 토대로 여러가지 방향에서 실험을 했다.

<br/>


Image representation을 위한 text supervision은 기존의 vision benchmark에서 엄청 낮은 성능을 보인다.

<br/>


텍스트의 일반성을 통해 더 광범위한 시각 개념을 익히는데 도움을 준다

<br/>
> ConVIRT trained from scratch, which we call CLIP

동등한 성능의 supervised ImageNet Model보다  zero-shot CLIP이 더 좋다. 즉 zero-shot performance가 해당 모델의 능력을 더 잘 보여준다고 생각할 수 있다.

<br/>


## 2. Approach

### 2.1 Natural Language Supervision

<br/>


자연어를 통한 비전의 지도학습은 새로운 기술이 아니고 이미 연구된 분야이다. 하지만 각 논문에서 다른 용어로 쓰일 때가 많다.

<br/>

Language supervised을 사용한 접근들의 핵심은 language를 훈련 신호로써 사용한다는 것이다. 다른 논문에서 사용된 특정 방식의 세부사항은 크게 중요하지 않다.

<br/>

Leraning from language는 확장이 쉽다는 장점이 존재한다. 라벨링할 필요가 없고, 인터넷에서 크롤한 방대한 데이터를 이용하여 훈련하기 때문이다.

<br/>

다른 장점은 learning from language는 representation을 학습하는 것 뿐만 아니라 representation과 language을 연결하며 학습한다는 것이다. 이로써 유연한 zero-shot transfer가 가능하다. => language의 일반성을 받기에

<br/>


### 2.2 Creating a Sufficiently Large Dataset

<br/>

Language supervision에 사용되던 기존 데이터셋이 3가지 있다.
그 중 2가지는 높은 퀄리티의 crowd-labeled 데이터셋이지만 현대의 기준에 비쳐보아 적은 수의 이미지를 가지고 있다.
나머지 하나는 데이터셋의 수는 많지만 메타데이터 정보가 부족하거나, 데이터셋 퀄리티가 일관적이지 않다. 많은 데이터 셋의 캡션이 이미지와 관계가 없어, 이를 필터링 한 결과 1/6으로 줄어드는 결과를 보였다. 이는 imageNet의 사이즈와 비슷하다.

<br/>

Language supervision의 출발점은 인터넷에서 사용 가능한 대규모 데이터 셋이다. Crowd-labeled 데이터는 대규모의 데이터를 포함하기 어렵고 그런 데이터셋도 없다.

<br/>

그냥 모든 데이터셋이 다 맘에 안든다. Language Supervision의 시작점이 language model마냥 방대한 양의 인터넷 데이터를 쓸려고 출발한 분야인데 이미 존재하는 데이터 양이 맘에 안든다

<br/>

그리하여 우리는 4억 개의 인터넷 image-text 데이터를 모았다. 다양한 visiual Concept을 위하여, 50만개의 쿼리 중 거기에 포함된 텍스트에 대한 image-text 데이터를 모은다.

<br/>

Base query list는 wikipedia에서 적어도 100번 이상 나온 쿼리로만 이루어져있다.

쿼리당 2만개 이상의 image-text 데이터가 균형있게 분포돼 있을 것이라고 추정한다.
이는 GPT-2를 훈련한 데이터셋의 수와도 맞먹는다. 위에서 설명한 데이터셋을 앞으로 WIT라고 명명하겠다.

<br/>

### 2.3. Selecting an Efficient Pre-Training Method
> contrastive learning을 시작한 이유

<br/>


SOTA computer vision은 계산을 겁나게 많이 한다.  이전 모델들은 제한된 ImageNet의 class만을 예측하는데 그 정도 썻다. 우리는 zero-shot을 위해 open set, 즉 제한되지 않은 class를 predict하기 때문에 직관적으로 보면 어려워 보인다.

<br/>

그래서 우리는 훈련 효율성의 키워드가 scaling language supervision임을 알았고 이를 기반으로 Pre-training Method을 선택했다.

<br/>

초기 접근은 VirTex와 같이 image CNN과 text transfomer를 from scratch 부터 동시에 훈련했다.
결과는 좋지 않았다. 훈련 속도가 너무 느려!

<br/>

이전의 접근은 각 이미지에 해당하는 exact word를 예측한다. 이는 이미지에 딱 맞는 텍스트와 코멘트, description이 방대하기에 어렵다.

<br/>

최근에 contrastive objectives 가 equivalent predictive objective 보다 효과적으로 representation을 학습한다는 연구가 있었다.

<br/>

    contrastive objectives  <=> equivalent predictive objective   대응되는 두 훈련방식 

우리는 이미지에 대한 정확한 단어를 예측하도록 하는 것이 아닌 더 쉬운 contrastive training system에 대해 찾도록 하겠다.

<br/>


N size 배치가 주어졌을때, n*n Pairings 로 훈련하게 된다. CLIP은 Multimodal embedding space를 훈련한다.

<br/>

1. Text/Image encoder는 Text/Image Embedding의 코사인 유사도가 최대가 되도록 훈련한다.
2. 그와 동시에 나머지 n^2 - n개의 unpaired target의 코사인 유사도가 최소가 되도록 text embedding을 훈련한다.

1,2 번을 반영하는 대칭 cross entropy를 최적화한다.

<br/>

데이터셋이 워낙 크다보니 overfitting에 대한 걱정은 필요 없다. Text/image encoder를 초기화 없이 from scratch부터 훈련한다. Image representation과 Contrastive embedding space간의 Non-linear projection말고, 각 Image/text representation을 동일한 multimodal space로 매핑할 linear projection만 사용한다.

<br/>

우리의 논문에 기반이 되는 Zhang et al.(2020)에 대해 text transformation function을 삭제하고, image transformation function을 간략화 한다.

<br/>


### 2.4 Choosing and Scaling a Model

<br/>

Image encoder는 ResNet-50과 Vit를 둘다 사용해봤다. 

<br/>

우선 ResNet-50는 광범위하게 사용되고 증명된 성능으로 인해 사용했다. 약간의 개선사항이 있다. Global average pooling layer를 attention pooling mechanism으로 바꿨다.
어텐션 풀링 레이어는 transformer의 multi-head QKV attention으로 구현되어 있다. 여기서 query는 이미지의 average-pooled representation으로 구성된다.

<br/>

두번째 Image encoder는 Vit로 했다. 위와 다르게 약간의 수정만 있다. Patch,  positional embedding 에 layer norm을 추가한다.

<br/>

Text encoder는 transformer를 사용한다.

<br/>

EOS 토큰에 대한 transformer의 최상위 layer의 activation은 layer norm되고, embedding space로 Linear project된 텍스트의 feature representation으로 사용된다. 

<br/>

이전의 computer vision research는 깊이나 넓이를 단독으로만 증가시켜 실험을 진행시켰다. 우린 깊이 넓이 resolution 동시에 그리고 동일하게 증가시켜 진행하겠다. 이전 연구에서 하나의 관점에 모든 스케일을 부담시키는 것보단 여러 측면에 균형있게 부담시키는 것이 더 좋은 성능을 거둔 다는 결과를 냈다.

<br/>

Text encoder는 넓이만 스케일하고 깊이는 그대로 둘 것이다. 

<br/>


### 2.5 Training

<br/>

>Image encoder : 5 ResNet, 3 Vit
>Text encoder : transformer

32 epoch, Adam optimizer, Batch size = 2^16


<br/>

## 3 Experiments

CLIP 모델은 이제 훈련하지 않는다. 중요한 것은 해당 밴치마크에 대한 직접적인 훈련없이 바로 사용된다. 그리고 해당 이미지 테스크에 맞춰 텍스트 인코더에 들어갈 텍스트를 적절히 생성해야한다. 

<br/>

예를 들어 강아지와 고양이를 분류하는 특정 테스크가 있다고 하자. 기존에는 이를 그냥 강아지,고양이에 대한 원핫 인코딩방식으로 각 데이터 포인트에 맞는 라벨 백터를 전달했다면, CLIP은 라벨 백터대신 텍스트 “a photo of a dog”와 같이 전달해야한다. 



<br/>

## summary and impression

이 논문의 핵심 키워드는 Contrastive learning이다. 하나의 정보에 대해서 supervision 방식으로 훈련하던 것을 버리고 n개의 묶음의 데이터를 가지고 맞는 것, 틀리는 것을 구분하며 훈련된다.

<br/>

또 하나의 큰 특징은 훈련시, 해당 dataset으로의 training이 필요없다는 점이다.

<br/>

VLMs의 기초가 되는 CLIP모델에 대해 읽어봤다. 크게 와닿은 점은 하나의 장난처럼 얘기되는 "scale all you need"이다. 그 이유는 이 CLIP의 성공의 이유는 엄청난 양의 데이터를 이용한 훈련이라고 생각하기 떄문이다. 최근 논문에서 사용되는 여러가지 모델들을 보면 하나같이 파라미터 수가 엄청나다. 

<br/>

scale이 커짐에 따라 생기는 문제들을, 여러가지 method나 architecture로 보조를 하는 듯한 연구도 의미가 있을 것 같다.
