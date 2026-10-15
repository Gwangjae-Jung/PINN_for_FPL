# Decision Log: Operator Learning for 2D Homogeneous FPL Equation using Separable PI-DeepONet

본 문서는 2차원 공간 균일(spatially homogeneous) Fokker-Planck-Landau (FPL) 방정식에 대한 Separable Physics-Informed Deep Operator Network (Sep-PI-DeepONet) 기반 오퍼레이터 학습 시스템을 설계 및 구현함에 있어 결정한 주요 아키텍처, 물리적 정식화, 데이터 생성 및 하이퍼파라미터 사항과 그 배경 근거를 기술합니다.

-----

## 1. 디렉터리 구조 및 명명 규칙

- **선택된 디렉터리명**: `fpl_separable_pi_deeponet/`
- **결정 배경**:
  - 기존 PyTorch 포팅 패키지인 `separable_pi_deeponet`과 PINN 학습 코드베이스인 `train/`, 수치 충돌 해석기인 `solver/`와 명확하게 상호작용하면서도, FPL 방정식 전용 오퍼레이터 학습 파이프라인을 독립적으로 유지 및 실행할 수 있도록 구성하였습니다.
  - 모듈별 역할 분담:
    - `config.py`: 문제 설정 및 하이퍼파라미터 정의 (`TypedDict` 기반)
    - `dataset.py`: Bi-Maxwellian 및 BKW 초기 조건 클래스 표본 생성기
    - `collision.py`: Fast Spectral Method (FSM) 충돌 연산자 및 거시적 보존량(질량, 운동량, 에너지) 평가기
    - `train_bkw.py`: BKW 해석해 클래스 대상 오퍼레이터 학습 스크립트
    - `train_bimaxwellian.py`: Bi-Maxwellian 클래스 대상 오퍼레이터 학습 스크립트
    - `evaluate.py`: 미학습 테스트 초기 조건에 대한 롤아웃 예측 및 오차·보존성 평가 스크립트

-----

## 2. 초기 조건 클래스(Classes of Initial Conditions)의 표본 공간 설계

오퍼레이터 학습(Operator Learning)은 단일 초기 조건이 아닌 함수 공간(function space) $\mathcal{U}$ 상의 다양한 초기 분포를 입력받아 해 궤적 $f(t, v)$로 매핑하는 비선형 연산자 $\mathcal{G}: f_0 \mapsto f$를 근사하는 것이 목적입니다. 이에 따라 두 가지 초기 조건 패밀리를 다음과 같이 매개변수화하여 연속적인 클래스를 구성하였습니다.

### (1) 두 Maxwellian 분포의 혼합 (Bi-Maxwellian Class)
- **물리적 배경**: 서로 다른 평균 속도와 온도를 가진 두 가스 성분이 충돌 완화를 거쳐 단일 평형 Maxwellian으로 수렴하는 대표적인 비평형 속도 공간 완화 과정.
- **수학적 정식화**:
  $$f_0(v) = \rho \left[ w_1 \mathcal{M}(v; u_1, \sigma_1) + w_2 \mathcal{M}(v; u_2, \sigma_2) \right]$$
- **기존 `train/` PINN 코드 참고 및 표본 분포 설정**:
  - 차원: $d = 2$
  - 절단 속도: $V_{\text{max}} = 5.0$ (`config/base_config__2d.py`의 `MAX_V__DICT['bimaxwellian'] = 5.0`)
  - 최대 시간: $T_{\text{max}} = 3.0$ (`config/base_config__2d.py`의 `MAX_T__DICT['bimaxwellian'] = 3.0`)
  - 총 질량 밀도: $\rho = 0.2$ (고정)
  - 중심 변위: $d_u \sim \mathcal{U}(0.6, 1.4)$, 방향각 $\theta \sim \mathcal{U}(0, 2\pi)$
    - $u_1 = (d_u \cos\theta, d_u \sin\theta)$, $u_2 = -u_1$ (계의 총 운동량이 0이 되도록 보장)
  - 모드별 온도/분산: $\sigma_1, \sigma_2 \sim \mathcal{U}(0.6, 1.0)$ (기존 `init_cond__std = 0.8` 중심)
  - 가중치: $w_1 \sim \mathcal{U}(0.4, 0.6)$, $w_2 = 1 - w_1$

### (2) BKW 해석해 클래스 (BKW Solution Class)
- **물리적 배경**: Maxwell 분자 상호작용($\gamma = 0$) 하에서 균일 Boltzmann 및 Landau 방정식의 엄밀한 비선형 비평형 해석해(Bobylev-Krook-Wu solution).
- **수학적 정식화**:
  $$K(t) = 1 - c_{\text{ext}} \exp\left(-2(d-1)\rho \kappa t\right)$$
  $$f_{\text{BKW}}(t, v) = \frac{\rho}{2\pi K(t)} \exp\left(-\frac{\|v\|^2}{2K(t)}\right) \left[ 2 - \frac{2 + \|v\|^2}{2K(t)} + \frac{\|v\|^2}{2K(t)^2} \right]$$
- **기존 `train/` PINN 코드 참고 및 표본 분포 설정**:
  - 절단 속도: $V_{\text{max}} = 2\pi \approx 6.28318$ (`MAX_V__DICT['bkw'] = 2*pi`)
  - 최대 시간: $T_{\text{max}} = 3.0$ (`MAX_T__DICT['bkw'] = 3.0`)
  - VHS 계수: $\kappa = 0.3125$, 지수: $\gamma = 0.0$
  - 총 밀도: $\rho = 0.2$
  - 초기 비평형 파라미터 표본 공간:
    $$c_{\text{ext}} \sim \mathcal{U}(0.1, 0.5)$$
    - **중요 결정**: $f_0(v) \ge 0$ (물리적 비음수성)을 전 속도 영역에서 보장하기 위한 필요충분조건은 $c_{\text{ext}} \le \frac{2}{d+2} = \frac{2}{4} = 0.5$입니다. 따라서 상한을 $0.5$로 엄격히 제한하여 물리적으로 타당한 비음수 분포군만을 생성하도록 하였습니다.

-----

## 3. Separable PI-DeepONet 아키텍처 및 좌표 분해 설계

- **좌표 축 분해**:
  시간 1차원 $t \in [0, T_{\text{max}}]$과 속도 2차원 $v_1, v_2 \in [-V_{\text{max}}, V_{\text{max}}]$의 총 3개 축에 대해 각각 1차원 독립 서브네트워크를 할당 ($d = 3$):
  - $\text{Trunk}_t(t): \mathbb{R}^1 \to \mathbb{R}^{r \cdot p}$
  - $\text{Trunk}_{v_1}(v_1): \mathbb{R}^1 \to \mathbb{R}^{r \cdot p}$
  - $\text{Trunk}_{v_2}(v_2): \mathbb{R}^1 \to \mathbb{R}^{r \cdot p}$
- **Branch 센서 인코딩**:
  - 속도 공간 $[-V_{\text{max}}, V_{\text{max}}]^2$ 상에 균일하게 배치된 $16 \times 16 = 256$개의 고정 센서 위치에서의 초기 분포값 $f_0(v_{\text{sensors}})$을 벡터화하여 Branch MLP의 입력으로 사용 ($m = 256$).
  - MLP 구성: $[256, 128, 128, 128, 100]$ (잠재 차원 $p = 100$).
- **텐서 랭크 $r$ 및 외적 연산**:
  - 텐서 랭크 $r = 50$, 잠재 차원 $p = 100$을 채택 (`separable-PI-DeepONet` 논문 및 Burgers/Heat 실험 설정 준용).
  - Einstein summation:
    - Trunk 결합: `'trpc, irpc, jrpc -> tijpc'`
    - Branch-Trunk 결합: `'bpc, tijpc -> btijc'`
    - 격자점 전체에 대한 순전파 결과의 형상은 $(N_{\text{batch}}, N_t, N_{v_1}, N_{v_2}, 1)$이 됨.
- **도함수 연산 차수 (`deg_grad`)**:
  - `deg_grad = [1, 0, 0]`로 설정하여 시간 $t$에 대한 1차 도함수만을 전방향 자동 미분(`torch.func.jacfwd` + `vmap`)으로 계산 및 캐싱.
  - 시간 편미분 $\frac{\partial f}{\partial t}$는 순전파 후 `compute_gradient([1, 0, 0])` 호출을 통해 $\mathcal{O}(1)$ 외적 연산으로 즉시 도출되어, 거대한 야코비안 행렬 역전파가 전혀 발생하지 않음.

-----

## 4. 손실 함수 정식화 및 물리적 제약조건

전체 목적 함수는 다음과 같은 복합 물리 정보 손실 함수로 구성됩니다:
$$\mathcal{L} = \lambda_{\text{IC}} \mathcal{L}_{\text{IC}} + \lambda_{\text{res}} \mathcal{L}_{\text{res}} + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}} + \lambda_{\text{pos}} \mathcal{L}_{\text{pos}}$$

1. **초기 조건 손실 ($\mathcal{L}_{\text{IC}}$)**:
   - $t=0$에서의 모델 예측값과 실제 초기 함수값의 오차:
     $$\mathcal{L}_{\text{IC}} = \frac{1}{N_{\text{batch}} N_v^2} \sum_{i=1}^{N_{\text{batch}}} \sum_{j, k} \left| f^{(i)}(0, v_{1, j}, v_{2, k}) - f_0^{(i)}(v_{1, j}, v_{2, k}) \right|^2$$
   - 가중치 $\lambda_{\text{IC}} = 20.0$ 부여 (초기 조건의 엄밀한 재현 보장).

2. **FPL 충돌 잔차 손실 ($\mathcal{L}_{\text{res}}$)**:
   - $\frac{\partial f}{\partial t} - Q(f) = 0$ 잔차를 평가:
     - **BKW 문제**: Maxwell 상호작용 하에서 $Q(f_{\text{BKW}}) = \frac{\partial f_{\text{BKW}}}{\partial t}$이므로, 해석적 미분 공식과의 잔차를 직접 평가하거나 스펙트럴 충돌 연산자와 비교.
     - **Bi-Maxwellian 문제**: `solver.fsm__fpl.FastSM_Landau_VHS`를 래핑한 `FPLSpectralCollision`을 활용하여 각 시간 단면에서의 정밀한 $Q(f)$를 산출한 뒤 잔차 계산.
   - 가중치 $\lambda_{\text{res}} = 1.0$.

3. **거시적 보존 법칙 손실 ($\mathcal{L}_{\text{cons}}$)**:
   - 기체 분자 운동론에서 충돌 불변량에 의해 질량 $\rho(t)$, 총 운동량 $P(t)$, 총 운동에너지 $E(t)$는 임의의 시간 $t$에서 보존되어야 함:
     $$\frac{d}{dt} \int_{\mathbb{R}^2} \begin{pmatrix} 1 \\ v \\ \|v\|^2 \end{pmatrix} f(t, v) \, dv = 0$$
   - 수치 적분을 통해 시간 $t=0$에서의 거시적 모멘트와의 차이를 손실로 부과:
     $$\mathcal{L}_{\text{cons}} = \text{MSE}(\rho(t) - \rho(0)) + \text{MSE}(P(t) - P(0)) + \text{MSE}(E(t) - E(0))$$
   - 가중치 $\lambda_{\text{cons}} = 5.0$.

4. **비음수성 손실 ($\mathcal{L}_{\text{pos}}$)**:
   - 입자 분포 함수 $f \ge 0$의 물리적 성질을 강제하기 위해 음수 성분에 페널티 부과:
     $$\mathcal{L}_{\text{pos}} = \text{mean}\left(\text{ReLU}(-f)^2\right)$$
   - 가중치 $\lambda_{\text{pos}} = 10.0$.

-----

## 5. 최적화 및 하이퍼파라미터 설정

| 항목 | 설정값 | 근거 |
|---|---|---|
| **Optimizer** | Adam | 요구사항 및 `separable-PI-DeepONet` 기본 옵티마이저 |
| **Initial Learning Rate** | $1.0 \times 10^{-3}$ | `train/train_parser.py` 및 `separable-PI-DeepONet` 표준값 |
| **LR Scheduler** | ExponentialLR ($\gamma = 0.95$, step = 200) | 안정적인 수렴을 위한 지수 감쇠 스케줄링 |
| **Number of Epochs** | 2,000 | 빠른 프로토타이핑 및 수렴 보장 (CLI 인자로 증감 가능) |
| **Batch Size** | 16 | 함수 공간 샘플링 및 GPU 메모리 적합 배치 크기 |
| **Train / Test Samples** | 200 / 50 | 클래스 내 다양한 모드 파라미터 커버리지 확보 |
| **Time Grid ($N_t$)** | 16 | $t \in [0, 3.0]$ 구간 분할 |
| **Velocity Grid ($N_v \times N_v$)** | $32 \times 32$ | Fast Spectral Method $Q(f)$ 고속 FFT 연산 및 해상도 타협점 |
| **Sensor Resolution** | $16 \times 16 = 256$ | Branch 입력용 속도 센서 격자 |
| **Activation** | Tanh | PINN 및 DeepONet 표준 완활 활성화 함수 |

-----

## 6. 요약

본 설계는 Separable PI-DeepONet의 가장 큰 장점인 **다차원 좌표 분해 및 초고속 외적 미분 계산**을 활용하여, 고차원 속도 공간 기체 분자 운동론 방정식인 Fokker-Planck-Landau 문제에 대해 단일 해 풀이를 넘어 **임의의 초기 조건 함수를 입력받아 시간 전개 해를 즉각 예측하는 신경망 연산자(Neural Operator)** 를 구축하였습니다.
