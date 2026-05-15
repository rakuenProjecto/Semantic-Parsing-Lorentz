# True Jacobian 연구 확장 구현 보고서

## 1. 작업 개요

이번 작업의 목적은 기존 `Semantic-Parsing-Lorentz` 코드베이스를 동적 곡률 Lorentz semantic parser에서 텍스트 임베딩 연구용 프로토타입으로 확장하는 것이었다. 변경 범위는 `Coor/Semantic-Parsing-Lorentz/` 내부로 제한했으며, 기존 기본 mock/dummy encoder 실행 경로는 유지하면서 새 기능은 설정값으로 선택 가능하게 추가했다.

추가된 연구 기능은 다음과 같다.

- 텍스트에서 문장 복잡도를 추정하는 lightweight linguistic complexity estimator
- CLS/text embedding을 입력 변수로 하는 true autograd Jacobian 계산
- true Jacobian에서 유도한 local metric `C_x = J_x J_x^T`
- Lorentz bilinear distance의 query-side local metric 보정
- Jacobian Frobenius norm, effective rank, 복잡도-곡률/Jacobian 상관 로깅
- JSONL 기반 실제 text classification / semantic parsing 스타일 데이터셋 지원
- tiny CPU smoke용 설정과 BERT true-Jacobian 실험용 설정

중요하게도, 구현된 main path는 learned Jacobian adapter 또는 `A_x = I + Delta` 방식이 아니다. `src/true_jacobian.py`에서 PyTorch autograd로 실제 Jacobian을 계산하고, 이 결과를 거리 계산에 전달한다.

## 2. 수정/생성된 파일 목록

`git status --short --untracked-files=all` 기준 변경 파일은 다음과 같다.

| 파일 경로 | 변경 내용 | 변경 이유 | 분류 |
| --- | --- | --- | --- |
| `Coor/Semantic-Parsing-Lorentz/src/complexity.py` | `ComplexityFeatures` dataclass와 `compute_linguistic_complexity` 구현. heuristic/spaCy 모드, raw/normalized/1~4 score 계산 포함. | 외부 NLP 모델 없이도 텍스트 기반 문장 복잡도 추정이 필요했기 때문. | Core utility |
| `Coor/Semantic-Parsing-Lorentz/src/true_jacobian.py` | CLS embedding 기준 true Lorentz/tangent Jacobian 계산, `C = J J^T`, Frobenius norm, effective rank 구현. | learned approximation이 아닌 실제 autograd Jacobian 기반 local metric을 만들기 위해 추가. | Core utility |
| `Coor/Semantic-Parsing-Lorentz/src/model.py` | `encode_cls`, `tangent_from_cls`, `curvature_from_cls`, `lorentz_point_from_cls` helper 추가. true Jacobian metric 옵션과 diagnostics output 추가. | Jacobian 계산 대상을 full encoder가 아니라 `cls_embedding -> Lorentz head`로 제한하고, distance에 local metric을 전달하기 위해 수정. | Core model code |
| `Coor/Semantic-Parsing-Lorentz/src/distance.py` | `LorentzBilinearDistance.pairwise`에 `query_local_metric` 인자 추가. `x_tilde = C_x x` 후 기존 bilinear scale 계산. clamp fraction scalar 저장. | `scale = (C_x x)^T W y` 형태의 true-Jacobian-derived local metric 보정을 지원하기 위해 수정. | Core distance code |
| `Coor/Semantic-Parsing-Lorentz/src/dataset.py` | mock dataset의 `complexity_mode`, `return_complexity_features` 지원. `JsonlSemanticParsingDataset` 추가. collate에서 `complexity_raw`, `complexity_normalized`, feature dict 지원. | synthetic complexity 외에 heuristic/spaCy complexity와 JSONL 데이터셋을 지원하기 위해 수정. | Dataset code |
| `Coor/Semantic-Parsing-Lorentz/train.py` | dataset/complexity/Jacobian 관련 CLI 추가. JSONL dataset builder, DDP local rank device 선택, Jacobian regularization/complexity loss, scalar logging, diagnostic save 옵션 추가. | 새 연구 기능을 학습 루프에서 선택적으로 활성화하고 scalar diagnostics를 기록하기 위해 수정. | Training code |
| `Coor/Semantic-Parsing-Lorentz/configs/default.yaml` | 새 dataset/complexity/true Jacobian 설정 기본값 추가. 기본값은 mock + synthetic + `use_true_jacobian_metric: false`. | 기존 기본 smoke 실행을 깨지 않으면서 새 기능을 config로 제어하기 위해 수정. | Config |
| `Coor/Semantic-Parsing-Lorentz/configs/bert_true_jacobian_complexity.yaml` | BERT encoder, heuristic complexity, true Jacobian metric, small batch 설정 추가. | UBAI/A10에서 BERT 기반 true-Jacobian 실험을 시작하기 위한 config. | Config |
| `Coor/Semantic-Parsing-Lorentz/configs/tiny_true_jacobian_smoke.yaml` | dummy encoder, `hidden_dim: 32`, `tangent_dim: 8`, batch 4, true Jacobian enabled 설정 추가. | CPU에서 true Jacobian 경로를 작게 검증하기 위한 smoke config. | Config |
| `Coor/Semantic-Parsing-Lorentz/tests/test_complexity.py` | simple sentence보다 complex sentence의 complexity score가 높은지 검사. | complexity estimator 기본 동작 검증용. | Test |
| `Coor/Semantic-Parsing-Lorentz/tests/test_true_jacobian.py` | tiny dummy model에서 true Jacobian forward, logits/Frobenius shape, finite 값 검사. | true Jacobian model path 최소 검증용. | Test |
| `Coor/Semantic-Parsing-Lorentz/tests/test_distance_true_jacobian_metric.py` | identity `query_local_metric`이 기존 pairwise distance와 거의 같은 결과를 내는지 검사. | distance 변경의 backward-compatible behavior 검증용. | Test |
| `Coor/Semantic-Parsing-Lorentz/scripts/smoke_true_jacobian.sh` | tiny true Jacobian config를 실행하는 shell script 추가. | 반복 가능한 smoke command 제공. | Script |
| `Coor/Semantic-Parsing-Lorentz/README.md` | complexity estimator, true Jacobian metric, JSONL format, 실행 명령, ablation table 문서화. | 새 연구 기능과 사용법을 문서화하기 위해 수정. | Documentation |
| `Coor/Semantic-Parsing-Lorentz/reports/true_jacobian_implementation_report.md` | 본 구현 보고서. | 변경 내용, 검증 상태, 실행 권장 사항을 기록하기 위해 생성. | Documentation/report |

참고: `git diff --stat -- Coor/Semantic-Parsing-Lorentz/`는 tracked file 6개만 집계하므로 untracked로 생성된 새 파일들은 stat 출력에 포함되지 않는다. 출력은 `6 files changed, 785 insertions(+), 70 deletions(-)`였다.

## 3. 핵심 구현 내용

### Linguistic complexity estimator

`src/complexity.py`가 새로 추가되었다. 핵심 API는 `compute_linguistic_complexity(text: str, mode: str = "heuristic") -> ComplexityFeatures`이다. heuristic 모드는 regex와 lexical rule만 사용하므로 offline smoke test에서 외부 모델 없이 동작하도록 설계되었다. spaCy 모드는 optional이며, `spacy` 또는 `en_core_web_sm`이 없으면 warning 후 heuristic으로 fallback한다. spaCy model은 `_SPACY_NLP`에 cache된다.

### True Jacobian computation

`src/true_jacobian.py`에서 `compute_true_lorentz_jacobian_wrt_cls`와 `compute_true_tangent_jacobian_wrt_cls`를 제공한다. 실제 구현은 `torch.autograd.grad`를 output dimension별로 반복 호출해 `[B, D, H]` 또는 `[B, T, H]` Jacobian을 만든다. `create_graph=True`일 때 Jacobian-derived metric을 통한 higher-order gradient가 가능하지만 비용이 크다. `vectorize` 인자는 현재 API 호환 목적으로 받지만 내부 구현은 full `[B, D, B, H]` Jacobian materialization을 피하기 위한 batched autograd loop이다.

### True-Jacobian-derived local metric

`jacobian_output_metric`은 `J`에서 `C = J J^T`를 계산한다. `normalize=True`이면 average trace로 정규화하고, `identity_mix`를 통해 identity matrix를 섞어 안정성을 높인다. 출력 shape은 `[B, D, D]`이다.

### Distance computation change

`src/distance.py`의 `LorentzBilinearDistance.pairwise`에 `query_local_metric: Optional[Tensor] = None`가 추가되었다. 제공되면 `x_tilde = torch.einsum("bij,bj->bi", query_local_metric, x)`를 계산하고 기존 `scale = torch.einsum("bd,de,ne->bn", x_tilde, metric, y)`를 사용한다. 제공되지 않으면 기존 `x`를 그대로 사용하므로 기본 경로는 유지된다.

### Training loss extensions

`train.py`는 기존 CE loss, metric regularization, curvature auxiliary loss에 더해 true Jacobian이 켜진 경우 다음 loss를 선택적으로 추가한다.

- `jacobian_reg_loss = mean(log1p(jacobian_frobenius)^2)`
- `jacobian_complexity_loss = smooth_l1(jac_norm_scaled, complexity_norm)`

두 항은 각각 `true_jacobian_reg_weight`, `true_jacobian_complexity_weight`가 0보다 클 때만 적용된다.

### Logging additions

학습 로그에는 기존 loss/accuracy 외에 CE loss, curvature loss, metric regularization, mean absolute curvature, mean complexity가 포함된다. true Jacobian이 활성화되면 Jacobian Frobenius mean/std, Jacobian regularization loss, Jacobian-complexity loss, complexity와 absolute curvature의 correlation, complexity와 Jacobian Frobenius norm의 correlation, diagnostics 요청 시 effective rank mean이 추가된다. full Jacobian tensor는 기본적으로 저장하지 않는다.

### New config files

`configs/default.yaml`은 기존 smoke 경로가 유지되도록 `use_true_jacobian_metric: false`, `complexity_mode: synthetic`, `dataset_type: mock`을 기본값으로 둔다. `configs/tiny_true_jacobian_smoke.yaml`은 CPU용 작은 true Jacobian smoke 설정이고, `configs/bert_true_jacobian_complexity.yaml`은 BERT + heuristic complexity + true Jacobian metric 실험용 설정이다.

### Tests and smoke script

세 테스트 파일이 추가되었다. complexity ordering, tiny model true Jacobian forward, distance identity local metric equivalence를 각각 검사한다. `scripts/smoke_true_jacobian.sh`는 tiny smoke config를 실행한다.

### README updates

README는 새 구조, complexity estimator, true Jacobian metric 수식, JSONL format, offline/true-Jacobian/UBAI 실행 명령, ablation study 표를 포함하도록 갱신되었다.

## 4. True Jacobian 구현 방식

계산되는 기본 Jacobian은 다음이다.

```text
J_x = d x / d h
```

여기서 `h`는 encoder가 출력한 CLS/text embedding이고 shape은 `[B, H]`이다. `x`는 `model.lorentz_point_from_cls(cls_embedding)`이 반환하는 query Lorentz point이며 shape은 `[B, D]`이다. `D = tangent_dim + 1`이다.

기본 target은 `true_jacobian_target: lorentz`이며 이때 출력 Jacobian shape은 다음과 같다.

```text
J_x: [B, D, H]
```

대체 mode인 `true_jacobian_target: tangent`에서는 다음 Jacobian을 계산한다.

```text
J_t = d tangent / d h
J_t: [B, tangent_dim, H]
```

tangent mode에서는 tangent-space metric을 만든 뒤 Lorentz coordinate metric의 spatial block에 삽입한다. 그러나 기본 config와 BERT config 모두 `lorentz` target을 사용한다.

이 구현은 learned approximation이 아니다. `src/true_jacobian.py`에서 autograd로 `cls_embedding -> tangent/curvature -> lorentz point` mapping의 실제 도함수를 계산한다. raw token ID나 전체 BERT input sequence에 대한 Jacobian은 계산하지 않는다. 이 설계는 full encoder Jacobian을 피하면서 Lorentz projection head에 대해서는 exact Jacobian을 얻기 위한 것이다.

local metric은 다음처럼 계산된다.

```text
C_x = J_x J_x^T
C_x: [B, D, D]
```

구현상 `torch.bmm(J, J.transpose(1, 2))`를 사용한다. 이후 optional trace normalization과 identity mixing을 수행한다.

거리 계산에서는 label side Jacobian을 만들지 않는다. label side는 기존 label Lorentz embedding을 그대로 사용하며, query side에만 local metric을 적용한다.

```text
scale = (C_x x)^T W y
```

코드상으로는 먼저 `x_tilde = C_x x`를 만들고, 기존 bilinear distance의 `einsum("bd,de,ne->bn", x_tilde, W, y)`를 호출한다. `query_local_metric`이 없으면 기존 `scale = x^T W y` 경로와 동일하다.

## 5. 문장 복잡도 계산 방식

heuristic complexity는 외부 NLP 모델 없이 다음 feature들을 regex/lexical rule로 계산한다.

- token count
- likely verb count: auxiliary 목록과 `-ed`, `-ing` suffix
- estimated verb argument count: verb 주변 noun-like token, prepositional complement, complement marker, wh-complement marker 기반 근사
- noun count: function word, verb/adjective/adverb candidate를 제외한 noun-like token
- adjective count: `-ive`, `-al`, `-ous`, `-ful`, `-less`, `-able`, `-ible`, `-ic`, `-ical`
- adverb count: `-ly`
- preposition count
- conjunction count와 subordinate marker count
- named entity count: non-initial capitalized token/span 기반 cue
- punctuation count: comma, semicolon, colon, parentheses, question mark

raw score는 구현에서 다음 가중합으로 계산된다.

```text
raw_score =
  0.10 * token_count
  + 0.40 * verb_count
  + 0.60 * verb_argument_count
  + 0.25 * adjective_count
  + 0.25 * adverb_count
  + 0.30 * preposition_count
  + 0.35 * conjunction_count
  + 0.45 * subordinate_marker_count
  + 0.20 * named_entity_count
  + 0.10 * punctuation_count
```

정규화는 saturating transform으로 수행된다.

```text
normalized_score = raw_score / (raw_score + 10.0)
```

그 뒤 `[0, 1]`로 clamp하고, 최종 complexity는 다음 범위로 매핑된다.

```text
complexity_1_to_4 = 1.0 + 3.0 * normalized_score
```

학습에서는 `batch["complexity"]`가 curvature auxiliary loss의 target을 만들 때 사용된다. true Jacobian mode에서 `true_jacobian_complexity_weight > 0`이면 complexity를 `[0, 1]`로 정규화하고 Jacobian Frobenius norm의 scaled 값과 `smooth_l1_loss`로 맞추는 Jacobian-complexity loss에도 사용된다.

## 6. 학습 과정 변화

기존 loss 구조는 다음과 같았다.

```text
loss = CE
     + metric_reg_weight * metric_regularization
     + curvature_aux_weight * curvature_loss
```

현재 구현은 true Jacobian이 켜졌을 때 다음 항을 선택적으로 추가한다.

```text
jacobian_reg_loss = mean(log1p(jacobian_frobenius)^2)
loss += true_jacobian_reg_weight * jacobian_reg_loss
```

또한 complexity와 Jacobian norm을 연결하는 항을 추가할 수 있다.

```text
complexity_norm = ((complexity - 1.0) / 3.0).clamp(0, 1)
jac_norm = jacobian_frobenius / mean_detached(jacobian_frobenius)
jac_norm_scaled = clamp(jac_norm, 0, 3) / 3
jacobian_complexity_loss = smooth_l1(jac_norm_scaled, complexity_norm)
loss += true_jacobian_complexity_weight * jacobian_complexity_loss
```

curvature-complexity auxiliary loss는 기존 구조를 유지하되, `complexity_mode: heuristic` 또는 `spacy`일 때 dataset이 계산한 텍스트 기반 complexity를 사용한다. `complexity_mode: synthetic`일 때는 기존 mock dataset의 synthetic complexity를 사용한다.

로깅에는 다음 scalar가 추가되었다.

- CE loss
- curvature loss
- metric regularization
- mean absolute curvature
- mean complexity
- mean/std Jacobian Frobenius norm
- Jacobian regularization loss
- Jacobian-complexity loss
- complexity와 absolute curvature의 correlation
- complexity와 Jacobian Frobenius norm의 correlation
- diagnostics 요청 시 mean effective rank

`--save_jacobian_diagnostics`가 켜진 경우에만 작은 batch의 full Jacobian과 local metric을 저장한다. 기본값은 false이다.

## 7. 실행 및 검증 결과

이번 보고서 작성 전 다음 상태 점검 명령을 실행했다.

```bash
git status --short --untracked-files=all
git diff --stat -- Coor/Semantic-Parsing-Lorentz/
git diff -- Coor/Semantic-Parsing-Lorentz/
git diff --check -- Coor/Semantic-Parsing-Lorentz/
```

`git diff --check -- Coor/Semantic-Parsing-Lorentz/`는 exit code 0으로 통과했다. 출력에는 Windows 환경에서 Git이 표시하는 CRLF normalization warning만 있었다.

```text
warning: in the working copy of 'Coor/Semantic-Parsing-Lorentz/...', LF will be replaced by CRLF the next time Git touches it
```

Python 기반 검증은 이 환경에서 수행하지 못했다. `python --version`을 실행하면 Python 코드가 시작되기 전에 다음 오류가 발생한다.

```text
'python.exe' 프로그램을 실행하지 못했습니다. 지정한 로그온 세션이 없습니다. 이미 종료되었을 수도 있습니다
CategoryInfo          : ResourceUnavailable
FullyQualifiedErrorId : NativeCommandFailed
```

따라서 `pytest`, compile check, smoke training은 실제로 실행되지 않았다. 이전 검증 시도에서도 `python -m compileall src train.py`, `python -m pytest tests`가 같은 Windows logon-session 오류로 실패했다. 이 문제는 프로젝트 코드 오류가 아니라 현재 실행 환경에서 Python launcher가 동작하지 않는 문제로 보인다.

Python이 정상 동작하는 로컬 또는 UBAI 환경에서 다음 명령을 권장한다.

```bash
python -m pytest tests
python train.py --epochs 1 --num_train_samples 32 --num_val_samples 16 --batch_size 8
python train.py --config configs/tiny_true_jacobian_smoke.yaml
```

## 8. UBAI에서 실행할 때 권장 명령어

A10 GPU 4장 기준으로는 일반 BERT fine-tuning보다 작은 batch size에서 시작하는 것이 안전하다.

```bash
torchrun --nproc_per_node=4 train.py \
  --config configs/bert_true_jacobian_complexity.yaml \
  --no_dummy_encoder \
  --encoder_name bert-base-uncased \
  --batch_size 4 \
  --output_dir outputs/ubai_bert_true_jacobian_complexity
```

true Jacobian은 `create_graph=True`일 때 higher-order derivative graph를 유지하므로 계산량과 GPU memory 사용량이 크다. 일반 BERT fine-tuning batch size를 그대로 쓰면 OOM 위험이 있다. 전체 BERT training 전에 반드시 아래 순서로 확인하는 것을 권장한다.

1. `python train.py --config configs/tiny_true_jacobian_smoke.yaml`
2. 작은 BERT batch, 예: `--batch_size 2` 또는 `--batch_size 4`
3. loss/logging 값이 finite인지 확인
4. 필요하면 `true_jacobian_identity_mix`, `true_jacobian_reg_weight`, `detach_true_jacobian_metric` ablation 수행

## 9. 예상 위험 요소 및 주의점

- `true_jacobian_create_graph: true`는 Jacobian metric을 통한 gradient flow를 가능하게 하지만 higher-order derivative 비용이 발생한다.
- BERT encoder와 함께 사용할 때 batch size가 크면 GPU memory OOM 위험이 높다.
- Lorentz distance는 `acosh` domain 제약이 있으므로 numerical stability가 중요하다. 기존 `stable_acosh` clamp를 유지했고 pairwise path에서 clamp fraction을 scalar로 보관하지만, 학습 중 clamp 비율이 높으면 metric/curvature 설정을 점검해야 한다.
- heuristic complexity는 gold syntactic parse가 아니다. verb argument count, noun/adjective/adverb 판정, named entity count는 모두 lexical cue 기반 근사다.
- true Jacobian metric이 성능 향상으로 이어지는지는 보장되지 않는다. 반드시 ablation study가 필요하다.
- `detach_true_jacobian_metric: true`는 real Jacobian을 계산하지만 `C_x`를 detach하는 ablation이다. main training mode는 detach하지 않는 경로다.
- `vectorize` 인자는 현재 실제 vectorized implementation이 아니라 API 호환 인자다. 현재 구현은 output dimension loop를 사용해 per-batch Jacobian을 만든다.

## 10. 추천 ablation study

| 실험 | Encoder | Lorentz curvature | Complexity loss | True Jacobian metric | Detached `C_x` | 목적 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline BERT classifier | BERT | No | No | No | No | 표준 classifier 기준선 |
| Lorentz curvature only | BERT/dummy | Yes | No | No | No | 기존 dynamic-curvature 효과 측정 |
| Lorentz curvature + complexity | BERT/dummy | Yes | Yes | No | No | complexity-curvature auxiliary 효과 측정 |
| Lorentz curvature + true Jacobian metric | BERT/dummy | Yes | No | Yes | No | local metric correction 단독 효과 측정 |
| Lorentz curvature + true Jacobian metric + complexity | BERT/dummy | Yes | Yes | Yes | No | 전체 연구 prototype 효과 측정 |
| Detached true Jacobian metric | BERT/dummy | Yes | Optional | Yes | Yes | `C_x` higher-order gradient path ablation |
| Different `identity_mix` values | BERT/dummy | Yes | Optional | Yes | No | metric stabilization 강도 sweep |

## 11. 결론

구현상으로는 문장 복잡도 추정, JSONL dataset, true autograd Jacobian 계산, Jacobian-derived local metric, Lorentz bilinear distance 보정, 학습 loss/logging 확장, smoke/BERT config, tests, README 문서화가 완료되었다. 기본 config는 `use_true_jacobian_metric: false`와 `complexity_mode: synthetic`이므로 기존 offline dummy encoder smoke 경로를 유지하도록 설계되어 있다.

다만 현재 Windows 실행 환경에서는 `python.exe`가 logon-session 오류로 실행되지 않아 Python 기반 검증을 수행하지 못했다. 실제 검증은 Python이 정상 동작하는 로컬 환경 또는 UBAI 환경에서 `pytest`, 기본 smoke run, tiny true-Jacobian smoke run 순서로 진행해야 한다. 특히 BERT true-Jacobian 실험은 작은 per-GPU batch size에서 시작하고, memory 사용량과 finite loss/logging 값을 먼저 확인해야 한다.
