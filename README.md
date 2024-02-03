# VLIS: Unimodal Language Models Guide Multimodal Language Generation

## Abstract 

VLM은 좋다. 하지만 VLM은 복잡한 언어적 이해를 배워야한다. 이 작업이 어렵다. 그래서 이 언어 이해 능력을 향상시키는 방식인 VLIS를 소개한다.  

이 기술은 추가 훈련없이 VLM의 visual conditioning capability와 language model의 language understanding을 결합한다. VLM의 point-wise mutual infomataion을 뽑아 내고, 뽑아낸 정보를 language model의 Token likelihood를 조정하기 위해 improtance sampling weight로 사용한다.  

## Introduction

대부분의 VLM은 LM에서 확장되었지만 LM의 전체적인 언어적 이해를 완전히 상속받지 못한다.  
2가지의 실패 예시가 존재한다.
1. 명명된 엔티티를 지정하지 않는다. (고유명사를 피한다) 이 문제는 knowledge의 부족에 의한 것이 아니다.  VLIS zero-shot method를 사용하니 괜찮아졌다.
2. Image context에 너무 의존한다. 그러지 말아야 할 경우에도, 이 결과로 질문이 말이 안되도, 모델이 상식을 벗어나 이미지에만 집중해 답을 만들어 낼 수 있다. 

따라서 VLM의 언어적 능력은 완벽하지 않다. Only language model은 이미 많은 실험에서 언어적 능력이 보장됐다. 그래서 VLM의 언어적 책임을 only language model에게 부담시키는 시도를 하겠다.  
우리가 제시한 VLIS method은 VLM의 언어적 이해력을 강화한다.  

Text token을 만들 때, Language model에서 만들어진 token Likelihood와 VLMs에서 추출한 importance sampling을 따른다.  

VLM의 visual conditioning capability를 분리하기 위해(언어적 능력은 제외하고 추출), exponental PMI(point-wise mutual infomation)를 사용한다. 위에서 구한 값과 현재 text token likelihood로 통합한다.  

이런 식으로 진행한다면, language model의 language modeling capability를 유지하며 visual conditioning 을 컨트롤 할 수 있다.(language model과 VLM의 장점만 사용할 수 있다)  

<br>

## 2. VLMs as Importance Sampling Weights

다시말하지만 VLIS는 VLM의 visual conditioning capability와 Language model의 언어적 능력가 조화를 이루게 하는 기술이다.

<br>


### 2.1 Intuition

VLM의 Pvl를 만들기 위해, text-only LM을 (이미지,텍스트) 데이터 셋을 가지고 MLE object function으로 fine-tuning한다. (LM만으로 VLM을 만든다는 얘기가 아니다. LM이 VLM에 부품으로 들어갔을때, 이러하다 라는 얘기다)  

하지만 대부분의 VLM의 object function을 보면 image conditioning likelihood를 최대화하는 것만 생각하고, 이는 특정 이미지에 의존하지 않은(상관없는) marginal likelihood Pvl(xt)를 만들고, 이를 통해 예기치 않은 출력을 만들 수 있다.  

이렇게 너무 이미지에만 집중하면, VLM은 훈련 데이터에 나타난 사회적 편견 반영,증폭하고, 또는 기존의 LM에 학습된 commonsence knowledge를 왜곡할 수 있다.  

요약하면, 기존에 VLM의 object function이 너무 image-based라 이런 방식으로 훈련하고, 이는 기껏 가져온 LM의 능력을 잘 못쓴다. 그래서 개선하겠다. 

<br>

### 2.2 Extracting visual Weights

일단 VLM에서 visual coditioning strength를 추출해, improtance sampling weight로 쓸 것 이다.  
PMI(point-wise Mutual infomation)을 쓸 것 이다. 이는 text,image간의 연관 정도를 측정하는 도구이다. 
PMI 의 방식을보면, 이전 토큰이 주어지면 다음 token과 image context간의 PMI를 측정한다.  

구해진 association을 보면 분자는 VLM의 image conditional likelihood로 쉽게 구할 수 있다.  
하지만 분모는 image context c의 marginalization가 필요하다. ( VLM에서의 text token에 대한 값만을 요하므로 image 관련 정보를 marginalization하는 것이다)  
> marginalization이란 하나의 확률 변수를 없애는 과정 (P(A,B) 에서 B의 probility를 통합=> P(A,b1) + P(A,b2) +... +  P(A,bn))
 
기존대로 진행한다면 하나의 PMI를 만들때 모든 이미지에 대한 marginalization이 필요하다. 엄청난 양의 계산이 필요하다.  
다음 3개의 프록시는 PMI의 방대한 계산 비용을 줄이기위한 방식이다.  

<br>

#### 2.2.1 Approximating the marginal

첫 시도는 text-only LM을 image-text 훈련 데이터로 따로 학습시키는 것이다. (VLM에서 marginalization하지 않고 그냥 LM에서 만들어서 사용하자)  
하지만 이 방식은 엄청 오래 걸릴것이다. 그리고 새롭게 훈련된 LM이 정확하게 marginal likelihood 을 내온다고 확신할 수 없다. 다른 모델을 훈련하는데 필요한 추가적인 복잡성에 의해  

두번째, 접근은 pre-selected image set의 평균을 전체 평균처럼 사용하는 것이다.  

마지막은, 시각적 정보가 없는 1~2개의 이미지에 대한 score로 훈련하는 것이다.  

우리는 계산비용을 줄이기 위해 3번째 방식을 선택한다.
우리는 2개의 시각정 정보가 없는 이미지 black/white-filled image를 image set으로 사용한다. 이 작은 이미지 셋으로 구해진 값으로 PMI의 분모인 marginal likelihood를 대채한다.  

이런 방식을 하면, VLM에서의 3번의 forward pass가 일어난다.
한번은 image-text의 conditional likelihood(분자)를 구하는 것, 2개는 위의 white/black - text을 사용한 marginal likelihood(분모)를 위한 것
white/black image - text 의 marginal likelihood를 더하고 반으로 나눈 것을 PMI의 분모로써 사용한다.  

그리고 text-only model에서도 한번의 forwardpass가 있다. (language generation 과정이 필요하기 때문)  

이후 우리는 image set의 다른 셋을 구성해, 특정 이미지 셋이 generation quality와 inference time에 합리적인 균형에 영향을 주는 것을 보일 것이다.  

<br>

### 2.3 Computing VLIS Score

모델의 진행과정은 다음과 같다. 
1. 우선 token likelihood를 LM으로부터 구한다. Ptext(Xt | c, X<t)  여기다가 추가로 language temperature에 대한 파라미터를 추가해 계산
2. 1번의 결과에 exponentiated PMI를 곱한다. 

이러면 VLIS score를 구할 수 있고, 다음 text token은 이 점수을 통해 예측한다.  

improtance sampling 방식은 이전에 연구된 논문에서 가져와, 현재 VLM, text-only LM에 마춰서 요소를 바꿨다.  
기존에는 (estimated quantity, nominal distribution, importance distribution) 간의 관계였지만 이 논문에선 (LM likelihood, VLM image-conditioned likelihood, importance distribution) 으로 치환되어 사용함.  

Approximating the marginal 2번에서 말한대로, 전체 이미지 데이터에 대한 expectation값이 아닌 일부 데이터만을 가지고 Pvl(xt|x<t)의 marginalization을 진행하겠다.  

따라서 이러한 방식들로 VLIS는 효울적으로 진행할 수 있다.  

<br>


#### 2.3.1 Fluency masking

PMI의 식에서 보듯이 분모에 해당하는 marginal likelihood Pvl(xt|x<t) 가 극단적으로 작은 값이 된다면, PMI값이 망가지고, 이에 따라 좋지않은 결과를 낸다.  

이런 이상한 결과를 막기위해, fluency mask라는 방식을 VLIS score를 구하는 식에 도입한다.  
text-only model에서 생성된 Ptext(xt|c,x<t)의 값이 임계치 보다 작다면 다음 토큰의 후보에서 제거한다. 

이제 결과가 나온 VLIS score를 보고 가장 점수가 높은 token이 다음 text로 예상이 된다.  

<br>

## summary

이 논문은 VLM의 약한 언어적 능력을 극복하기 위한 method를 연구한 논문이다. 

language model에서 나온 Token 을 베이스로 해서, VLM의 visual weight를 PMI를 통해 추출한다. 이 둘을 곱해서 도출된 VLIS score를 통해 다음 token을 우추하게 된다.  

그리고 PMI의 전통적인 단점인 marginal likelihood가 작으면 값이 극으로 치우쳐지는 문제도 해결한 논문이다.
