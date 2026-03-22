# Phase 2: 병렬 코퍼스 구축

## 목표

USFM 파일에서 절 단위로 정렬된 영어-나마어 병렬 코퍼스를 만든다.
이 데이터가 번역 모델의 학습 데이터가 된다.

## 데이터 소스

### 나마어 (Nama, nmx)
- **출처 1**: eBible.org — 누가복음, 사도행전 (2권, 2,151쌍)
- **출처 2**: Paratext RTF → USFM 변환 — 신약 27권 전체

### 영어 (English)
- **World English Bible (WEB)**: eBible.org에서 다운로드, 구약+신약 전체
- Public Domain

### 다국어 (Multilingual)
- PNG 7개 언어: Hiri Motu, Tok Pisin, Bedamuni, Bine, Gizrra, Gidra, Magi
- png.bible에서 USFM 수집
- 나마어와 같은 지역의 언어들로, 다국어 전이 학습에 활용

### 백트랜슬레이션 (Back Translation)
- Paratext RTF에서 BT 파일 추출
- 나마어 → 영어 역번역 (나마어 화자가 작성)
- 26권, 2,622쌍

## 파싱 스크립트

```bash
# 영어-나마어 병렬 코퍼스
uv run python scripts/parse_usfm.py

# 다국어 병렬 코퍼스
uv run python scripts/build_multilingual.py
```

## 출력 파일

| 파일 | 크기 | 내용 |
|------|------|------|
| `nama_eng_parallel.json` | 3.2 MB | 영어-나마어 NT 전체 (3,167 정렬쌍) |
| `nama_bt_parallel.json` | 2.2 MB | 나마어-BT 26권 (2,622쌍) |
| `multilingual_parallel.json` | 59 MB | 7개 PNG 언어 병렬 |

## 데이터 구조

```json
{
  "ref": "MAT 1:1",
  "book": "MAT",
  "chapter": 1,
  "verse": 1,
  "nama": "Yéné sem Yesu Kerisoene...",
  "english": "The book of the genealogy of Jesus Christ...",
  "aligned": true
}
```

## 알려진 문제

- 일부 절이 멀티-절로 합쳐져 있음 (나마어 RTF 원본의 구조 문제)
- 길이 비율 이상치 존재 (나마어가 영어 대비 너무 길거나 짧은 절)
- → Phase 4에서 클리닝 처리
