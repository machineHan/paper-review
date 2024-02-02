# Fusing Pre-trained Language Models with Multimodal Prompts through Reinforcement Learning
<br>
## Before reead

논문에 들어가기 앞서, commonsense reasoning와 Maximum likelihood estimator, prompt engineering/tuning 에 대해 설명하겠다.  

`Commonsense Reasoning` 이란 아주 당연한 것에 대한 추론이다. 사람이 하는 행동에서 딱히 고민하지 않고 수행되는 것이다. 하지만 이를 기계가 인식하도록 하는데는 엄청난 수고가 든다.
> 예를 들어 우리는 물을 흘리면 큰 고민 없이 주변에 물을 흡수 가능한 소재로 바닥을 닦고 이를 버리거나 짜서 말린다. 하지만 이를 기계에게 실행하기에는 엄청난 수고가 든다.

이하 CSR로 줄여서 표기하겠다.  

`Maximum likelihood Estimator(MLE)` 는 likelihood가 최대인 모수를 찾아 최적화 시키는 방식에 일종이다.  
> 즉,  어떤 확률 분포에서 주어진 데이터가 있을 때, 데이터를 가지고  모수(파라미터)를 찾음

    Bayes’ rule : P(Y|X) = P(Y) * P(X|Y) / P(X)  
		=> posterior  = (prior * likelihood) / (const)  
    Posterior = feature가 주어졌을때, 특정 클래스일 확률  
    Likelihood = 클래스가 주어졌을때, 특정 feature일 확률  

e.g.  
MLE : Likelihood = P(X|Y)가 최대인 지점을 찾자, 관측결과에 민감하다!  
MAP : posterior = P(X|Y)P(X)가 최대인 지점을 찾자  


`Prompt engineering`이란 제공된 언어모델 인터페이스에서 “유저가 원하는 정확한 정보”를 위해 사전적으로 입력하는 입력 문장  
e.g.  GPT한테  질문하기전에, ‘너는 나의 머신러닝 선생님이야’ , ‘이런 구조로 알려줘’ 등등 먼저 입력하는 과정을 뜻함  

Prompt tuning이란 언어모델을 fine-tuning하긴 하는데 주어진 입력 프롬프트에 대해 원하는 답변을 생성하도록 언어모델을 국소적으로 fine-tuning한다.  
e.g. 질문에 대한 대답을 내가 원하는 대답을 하게끔 fine-tuning  
  
Fine-tunin은 downstream task를 위해 추가적인 대규모 훈련을 가해 특화된 모델을 만드는 과정  


  <br>
## Abstract

Language model은 CSR이 가능하다. 
Language model의 knowledge를 확장시켜 멀티모달 인풋에 대해서도 적용시켜 CSR 을 할 수 있을까?    

ESPER라는 기술로 가능하게 하겠다. 이 기술은 text로만 pretrain 된 모델을 가지고 multimodal task를 처리할 수 있다.  

핵심기술은 강화학습이다. Multimodal input을 언어 모델과 일치시킬 것인데 여기서 direct supervision방식이 아니라 강화학습이 사용된다.  

Reward optimization은 CLIP에서 파생된 코사인유사도로 결정하고, multimodal pair dataset(=labeling)이 따로 필요없다.  

ESP라는 새로운 데이터셋을 포함한 여러가지 benchmark에 대해서도 좋은 성능을 낸다.  

> ESP : 하나의 이미지에 대한 여러가지 text가 포함된 multimodal dataset

> ESPER = Extending Sensory PErception with Reinforcement learning, language model의 인지능력을 강화학습을 통해 multimodal task까지 확장시키는 Framework
  <br>
## Introduction

새로운 multimodal환경에서 데이터를 수집하는 건 어렵다. 
시간도 많이들고, 모은 데이터셋이 질이 낮을 수도 있다.  

다른 multimodal domain 마다 매번 다른 dataset pairing을 걸치지 않고 좋은 CSR을 하길 바란다.  

이를 해결하는 framework ESPER를 소개한다. 핵심은 “강화학습”으로 추가적인 “annotation”없이 다양한 multimodal input을 pre-trained language model에 사용할 수 있다.
이로써 Pretrained language model에 내재된 여러가지 정보를 다양한 multimodal capability로 확장 가능하다.    

ESPER는 keyword는 multimodal prompt tuning과 reinforcement learning이다.  

ESPER는 동결된 language 모델을 사용하고,  visual feature을 language model에 매핑하기 위해 최소한의 encoder parameter룰 훈련한다.  

하지만 기존의 multimodal prompt tuning방식과 다르게, ESPER는 maximum likelihood estimation대신, 강화훈련 방식으로 파라미터를 훈련한다. 그래서 specific domain annotation이 필요 없다 = Direct supervised가 없다.  

일부 훈련과정을 보면 먼저 visual feature를 만들고 , encoder는 proximal policy optimization(PPO, reinforce learning)을 사용해 두번째 CLIP에서의 image-caption 유사도가 최대가 되도록 파라미터를 업데이트한다.   

이 과정의 결과로 frozen language model은 추가적인 annotation과정 없이 multimodal input을 사용할 수 있게 된다.  

강화학습은 MLE에 비해 2가지 장점이 존재한다.  
첫째 annotation이 필요없다. 
둘 째, 강화학습은 일반화능력을 유지한다. 우리 과정중 forzen language model 을 사용하는데, 이 방식이 reasoning capacites를 유지시키는 데 큰 역할을 한다.  
우린 maximum likelihood prompt tuning과 decoding-time method를 통해 language model에 이미지를 넣는 이전의 두 기술과 비교해보겠다.   
  <br>
## 2. Method 

ESPER의 3가지 구성요소
1) CLIP’s non-generative image/text encoders
2) a left-to-right language generator such as GPT-2 or COMET
3) an encoder that projects multimodal inputs into the subword embedding space of the language generator

다시 한 번 강조하지만 CLIP이랑 Language generators의 파라미터는 동결된 채로 진행된다.  

backprop를 위해 대부분의 아키텍처를 지나지만 업데이트는 오직 encoder에서만 행해진다  

보상은 CLIP에서 생성된 이미지/텍스트 유사도에 의존한다.  
  <br>
## 2.1 Architecture
  <br>
### CLIP  

2개의 CLIP이 존재한다. CLIP-I/T  
CLIP-I는 fixed CLIP image encoder로 이미지로 부터 feature vector를 추출한다.  
CLIP-T는 fixed CLIP text encoder로 언어 모델이 생성한 텍스트에 대한 백터를 만든다  
마지막으로,  CLIP-I/T에서 나온 백터를 가지고 유사도를 가지고 강화학습을 진행한다.  
  <br>
### Encoder

ESPER에서 유일하게 훈련되는 부분이다.  
CLIP-I에서 나온 output을 입력으로 사용한다. k길이의 백터를 뱉어낸다.  
출력된 백터는 pre-trained language model의 입력으로 사용된다.  
  
인코더에서 나온 image representation은 multimodal prompt로써 사용된다. 그리고 출력이 text representation과 결합된다.  
  <br>
### Pre-trained Language Model

Autoregressive language model은 이전의 토큰들의 정보를 가지고 다음 토큰을 예측한다.  
  
우리는 prompt tuning in text-only domain에서 따와, 아까 Encoder에서 생성한 hj와 GPT2의 text embedding lookup layer의 출력값과 연결시켜 사용한다.  
  
쉽게 말하자면, 기존의 언어모델은 이전 토큰에 대한 값들에 대해서만 사용해서 다음 텍스트 토큰에 대해 계산했다. 
하지만 우린 encoder output과 위의 방식을 concate해서 likelihood를 계산한다.  
  <br>
## 2.2 Training
  <br>
### Reinforcement Learning
강화학습을 생성 텍스트 - 이미지에 대한 코사인 유사도를 가지고 행하겠다. 강화학습 관점에서 보면, 텍스트 생성모델은 Policy이다.  우리는 보상 모델로 PPO-clip을 사용하겠다. 그리고 RL policy와 기존의 policy 사이의 KL distance를 제한해 생성 텍스트의 퀄리티를 보장한다.  

Reaward는 논문에 정의된 식 4번과 같이 정의 된다. 여기서의 하이퍼파라미터 수치는 실험을 통해 얻었다.  
  <br>
### Language Model Stability
Reward hacking은 에이전트가 일관되지 않지만 큰 보상을 얻는 텍스트를 발견했을 때, 주로 발생한다. 즉, 성능에 악영향을 끼치는데 좋은 보상을 받는 데이터를 발견했을 때이다.  
훈련 과정을 안정시키기 위해 보조 reward를 사용한다.  
첫째, 우리 모델에서 나온 텍스트의 likelihood와 완전 쌩 GPT-2에서의 likelihood를 통해 KL distance를 구한다. 이를 ESPER모델의 텍스트 생성 능력을 유지한다. 그리고 반복되는 어구에 높은 likelihood를 잘 못 부여하는 일이 종종 발생하여, 이를 막기 위해 repetation  penalty를 만들었다. 
  <br>
## 2.3 Adaptation on Pretrained Language Model

Multimodal 뿐만 아니라 domain-specific에서도 가능하다.

  <br>
## summary
ESPER는 pretrained-language generator에 있는 지식의 language generation capability와 CLIP이 정렬시킨 multimodal to text를 supervision없이 결합시킨다. 이 과정에서 강화학습이 사용된다.  
ESPER의 multimodal prompt tuning을 통해서 pretrained language model의 진행 시킨 prompt쪽으로 확장시킬 수 있다.  






1. 이미지 준비
2. 이미지를 CLIP-I에 삽입
3. CLIP-I의 출력을 Encoder의 입력으로 삽입
4. Encoder의 출력과 text embedding lookup layer을 concatenation 한 후, GPT에 삽입 
=> 이렇게 하는 이유는 GPT-2가 auto_regressive model이기 떄문에 이전 출력에 대한 정보를 받아 오는 것이다.
5. GPT에서 생성된 text를 가지고 CLIP-T에 삽입
6. 2,5번에서 나온 출력을 가지고 코사인 유사도 계산
7. encoder를 코사인 유사도를 가지고 강화학습(PPO)

이러면 이미지를 통한 multimodal prompt 완료

Integrated
Infrastructure

