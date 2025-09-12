# Online Reinforcement Learning Research Framework

PyTorch 기반의 온라인 강화학습 알고리즘 연구 프레임워크입니다. 모듈화된 구조로 새로운 알고리즘을 쉽게 추가하고 비교 실험을 수행할 수 있도록 설계되었습니다.

## 주요 특징

- **모듈화된 디자인**: `algorithms`, `buffers`, `core` 등 기능별로 코드가 분리되어 있어 확장 및 유지보수가 용이합니다.
- **Hydra 기반 설정 관리**: `configs` 폴더 내의 `.yaml` 파일을 통해 실험의 모든 하이퍼파라미터를 관리하여 재현성을 높입니다.
- **Weights & Biases (W&B) 연동**: `wandb`를 통해 학습 과정, 결과, 동영상 등을 실시간으로 트래킹하고 시각화합니다.
- **Gymnasium 지원**: `gymnasium` 및 `gymnasium-robotics` 환경을 지원하여 다양한 환경에서 알고리즘을 테스트할 수 있습니다.

## 설치 방법

1.  **저장소 클론**
    ```bash
    git clone <your-repository-url>
    cd online_rl
    ```

2.  **가상환경 생성 및 활성화**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **필요 라이브러리 설치**
    > **참고**: `requirement.txt` 파일의 이름을 `requirements.txt`로 변경하는 것을 권장합니다.

    ```bash
    pip install -r requirements.txt
    ```

## 실행 방법

`main.py`를 실행하여 학습을 시작합니다. `hydra`를 통해 커맨드 라인에서 직접 설정을 변경할 수 있습니다.

-   **기본 실행**:
    `configs/config.yaml`에 정의된 기본값으로 `sac` 알고리즘을 `Walker2d-v5` 환경에서 학습합니다.

    ```bash
    python main.py
    ```

-   **알고리즘 및 환경 변경**:
    커맨드 라인에서 `algorithm`과 `env_id`를 직접 지정하여 실행할 수 있습니다.

    ```bash
    # TD3 알고리즘으로 HalfCheetah-v4 환경에서 학습
    python main.py algorithm=td3 env_id=HalfCheetah-v4

    # TQC 알고리즘으로 Ant-v4 환경에서 학습
    python main.py algorithm=tqc env_id=Ant-v4
    ```

-   **학습 재개**:
    `load_model=True`와 함께 체크포인트가 저장된 경로를 `load_checkpoint_dir`로 지정하여 학습을 재개할 수 있습니다.

    ```bash
    python main.py load_model=True load_checkpoint_dir=logs/sac/Walker2d-v5/2025-09-12_10-54-34/
    ```

모든 학습 결과(로그, 모델 체크포인트, 설정 파일)는 `logs/` 디렉토리 아래에 알고리즘, 환경, 타임스탬프 기반으로 자동 생성된 경로에 저장됩니다.

## 구현된 알고리즘

-   [SAC (Soft Actor-Critic)](algorithms/sac/)
-   [TD3 (Twin Delayed Deep Deterministic Policy Gradient)](algorithms/td3/)
-   [TD7](algorithms/td7/)
-   [TQC (Truncated Quantile Critics)](algorithms/tqc/)
-   [HRAC](algorithms/hrac/)

## 프로젝트 구조

```
/
├───main.py                 # 메인 실행 파일
├───requirements.txt        # 의존성 라이브러리 목록
├───algorithms/             # 강화학습 알고리즘 구현
├───buffers/                # 리플레이 버퍼 구현
├───configs/                # Hydra 설정 (.yaml) 파일
├───core/                   # Trainer, Logger 등 핵심 로직
├───envs/                   # 환경(Gymnasium) 래퍼
├───logs/                   # 학습 로그 및 모델 체크포인트 저장
└───utils/                  # 유틸리티 함수
```
