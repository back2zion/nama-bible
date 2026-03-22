# Phase 3: 모델 학습

## 모델 선정 과정

나마어는 극저자원 언어(ultra-low-resource)로, 대부분의 다국어 모델에 포함되지 않는다.

| 모델 | 장점 | 한계 | 채택 |
|------|------|------|------|
| NLLB-200 (Meta) | 200+ 언어 지원 | 나마어 미포함, 600M 파라미터 | v1 베이스라인 |
| TranslateGemma 4B | 번역 특화, QLoRA 가능 | 비교적 작은 모델 | v2 |
| TranslateGemma 27B | 번역 특화, 대형 모델 | 2x GPU 필요 | v3 (현재) |
| GPT-4o / Claude | 범용 능력 우수 | 나마어 생성 불가, API 비용 | 불채택 |
| MADLAD-400 | 400+ 언어 | 나마어 품질 미검증 | 불채택 |

**결론**: 극저자원 언어에서는 모델 크기보다 **데이터 품질**이 병목이다.
번역 특화 모델(TranslateGemma)을 파인튜닝하는 것이 현실적 최선.

## v1: NLLB-200 Baseline

- **모델**: facebook/nllb-200-distilled-600M
- **데이터**: Luke + Acts (2,151쌍)
- **GPU**: 1x RTX 3090
- **학습 설정**: Adafactor, batch 32, 20 epochs, FP16

| Metric | Zero-shot | Fine-tuned |
|--------|-----------|------------|
| BLEU   | 0.08      | 3.92       |
| chrF   | 13.39     | 33.27      |

## v2: TranslateGemma 4B

- **모델**: google/translategemma-4b-it + QLoRA
- **데이터**: Full NT 27 books (3,167쌍)
- **GPU**: 1x RTX 3090

## v3: TranslateGemma 27B

- **모델**: google/translategemma-27b-it + QLoRA (4-bit NF4 양자화)
- **데이터**: Full NT 27 books + multilingual
- **GPU**: 2x RTX 3090 (48 GB VRAM)

### QLoRA 설정

```python
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### 학습 하이퍼파라미터

```python
MAX_SEQ_LEN = 256
PER_GPU_BATCH = 2
GRAD_ACCUM_STEPS = 8        # effective batch = 32
LEARNING_RATE = 1.5e-4
EPOCHS = 5
WARMUP_RATIO = 0.05
OPTIMIZER = "paged_adamw_8bit"
LR_SCHEDULER = "cosine"
```

### 학습 가능 파라미터

- Trainable: 233M / Total: 27.6B (0.84%)
- 모델 로딩: ~17분 (12 shard)
- VRAM 사용: ~36 GB

## v3.1: 클리닝 데이터 재학습

- **데이터 변경**: 3,167 raw → 1,737 clean + 1,403 BT = 3,140쌍
- **결과**: chrF 33.27 → **59.77** (+26.50 향상)

핵심: 데이터 품질 개선만으로 chrF가 거의 2배 상승. 노이즈 제거가 모델 크기 증가보다 효과적.

## v3.2: 화자 피드백 반영 재학습 (진행 중)

- **데이터 변경**: 3,140 + 룻기 화자 번역 85쌍 = 3,225쌍
- **목표**: 구약 번역 품질 향상, 화자의 어휘/표현 패턴 학습

## 실행 방법

```bash
# 학습
uv run python scripts/train_v3.py

# 평가
uv run python scripts/evaluate_v3.py

# 번역 생성
uv run python scripts/translate_draft.py RUT
```
