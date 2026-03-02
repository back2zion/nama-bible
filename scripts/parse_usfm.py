#!/usr/bin/env python3
"""
Nama Bible USFM Parser & Parallel Corpus Builder
나마어 성경 USFM 파서 및 병렬 코퍼스 구축 도구
"""

import re
import json
import csv
from pathlib import Path

# ─── USFM 파서 ───────────────────────────────────────────────────────────────

def parse_usfm(filepath: str) -> dict:
    """
    USFM 파일을 파싱하여 {book: {chapter: {verse: text}}} 구조로 반환
    """
    result = {}
    current_book = None
    current_chapter = None
    current_verse = None
    verse_buffer = []

    def flush_verse():
        if current_book and current_chapter and current_verse:
            text = " ".join(verse_buffer).strip()
            text = clean_usfm_text(text)
            if text:
                result.setdefault(current_book, {}).setdefault(current_chapter, {})[current_verse] = text
        return []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 책 ID
            m = re.match(r"\\id\s+(\w+)", line)
            if m:
                verse_buffer = flush_verse()
                current_book = m.group(1)
                current_chapter = None
                current_verse = None
                continue

            # 장
            m = re.match(r"\\c\s+(\d+)", line)
            if m:
                verse_buffer = flush_verse()
                current_chapter = int(m.group(1))
                current_verse = None
                continue

            # 절
            m = re.match(r"\\v\s+(\d+)\s*(.*)", line)
            if m:
                verse_buffer = flush_verse()
                current_verse = int(m.group(1))
                rest = m.group(2).strip()
                verse_buffer = [rest] if rest else []
                continue

            # 마커 제거 후 텍스트 누적
            if current_verse is not None:
                # 마커 라인이 아닌 경우 버퍼에 추가
                if not line.startswith("\\") or re.match(r"\\[qmspf]", line):
                    cleaned = re.sub(r"\\[a-z0-9+*]+\s*", " ", line)
                    if cleaned.strip():
                        verse_buffer.append(cleaned.strip())

    flush_verse()
    return result


def clean_usfm_text(text: str) -> str:
    """USFM 마커 및 불필요한 태그 제거"""
    # 각주 블록 제거 (\f ... \f*)
    text = re.sub(r"\\f\s+.*?\\f\*", "", text)
    # 상호참조 블록 제거 (\x ... \x*)
    text = re.sub(r"\\x\s+.*?\\x\*", "", text)
    # Strong 번호 제거 (|strong="G1234" 형식)
    text = re.sub(r'\|strong="[A-Z]\d+"', "", text)
    # 인라인 마커 제거 (\wj ...\wj*, \add ...\add* 등)
    text = re.sub(r"\\[a-z0-9+*]+\*?", " ", text)
    # 꺾쇠 태그 제거
    text = re.sub(r"<[^>]+>", "", text)
    # 연속 공백 정리
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─── 병렬 코퍼스 구축 ────────────────────────────────────────────────────────

def build_parallel_corpus(nama_data: dict, eng_data: dict) -> list:
    """
    나마어-영어 병렬 코퍼스 구축
    Returns: [{"ref": "LUK 1:1", "nama": "...", "english": "..."}, ...]
    """
    corpus = []
    book_map = {"LUK": "LUK", "ACT": "ACT"}

    for book_code in ["LUK", "ACT"]:
        nama_book = nama_data.get(book_code, {})
        eng_book = eng_data.get(book_code, {})

        all_chapters = sorted(set(list(nama_book.keys()) + list(eng_book.keys())))

        for chapter in all_chapters:
            nama_ch = nama_book.get(chapter, {})
            eng_ch = eng_book.get(chapter, {})
            all_verses = sorted(set(list(nama_ch.keys()) + list(eng_ch.keys())))

            for verse in all_verses:
                ref = f"{book_code} {chapter}:{verse}"
                entry = {
                    "ref": ref,
                    "book": book_code,
                    "chapter": chapter,
                    "verse": verse,
                    "nama": nama_ch.get(verse, ""),
                    "english": eng_ch.get(verse, ""),
                    "aligned": bool(nama_ch.get(verse) and eng_ch.get(verse))
                }
                corpus.append(entry)

    return corpus


# ─── 언어학적 분석 ────────────────────────────────────────────────────────────

def analyze_nama_linguistics(corpus: list) -> dict:
    """나마어 기초 언어학적 패턴 분석"""
    nama_verses = [e["nama"] for e in corpus if e["nama"]]

    all_words = []
    for v in nama_verses:
        words = re.findall(r"[a-zA-ZÀ-ÿéáóúíñü]+", v, re.UNICODE)
        all_words.extend([w.lower() for w in words])

    # 단어 빈도
    from collections import Counter
    word_freq = Counter(all_words)

    # 평균 단어 수 / 절
    avg_words = sum(len(re.findall(r"\S+", v)) for v in nama_verses) / len(nama_verses) if nama_verses else 0

    # 고유 문자 셋
    all_chars = set("".join(all_words))
    special_chars = sorted([c for c in all_chars if ord(c) > 127])

    # 접미사 패턴 (나마어는 교착어적 특성 예상)
    suffixes = Counter()
    for w in all_words:
        if len(w) >= 4:
            suffixes[w[-3:]] += 1
            suffixes[w[-2:]] += 1

    return {
        "total_verses": len(nama_verses),
        "total_words": len(all_words),
        "unique_words": len(word_freq),
        "avg_words_per_verse": round(avg_words, 2),
        "top_50_words": word_freq.most_common(50),
        "special_characters": special_chars,
        "top_20_suffixes": suffixes.most_common(20),
        "type_token_ratio": round(len(word_freq) / len(all_words), 4) if all_words else 0,
    }


# ─── 메인 실행 ────────────────────────────────────────────────────────────────

def main():
    base = Path(__file__).resolve().parent.parent
    data = base / "data"

    print("=" * 60)
    print("  나마어 성경 코퍼스 구축 파이프라인")
    print("=" * 60)

    # 1. 나마어 파싱
    print("\n[1/4] 나마어 USFM 파싱 중...")
    nama_data = {}
    for usfm_file in sorted((data / "nmx_ebible").glob("*.usfm")):
        parsed = parse_usfm(str(usfm_file))
        nama_data.update(parsed)
        for book, chapters in parsed.items():
            total = sum(len(v) for v in chapters.values())
            print(f"  ✓ {book}: {len(chapters)}장, {total}절")

    # 2. 영어 파싱 (Luke, Acts만)
    print("\n[2/4] 영어(WEB) USFM 파싱 중...")
    eng_data = {}
    for usfm_file in sorted((data / "eng").glob("*.usfm")):
        parsed = parse_usfm(str(usfm_file))
        eng_data.update(parsed)
        for book, chapters in parsed.items():
            total = sum(len(v) for v in chapters.values())
            print(f"  ✓ {book}: {len(chapters)}장, {total}절")

    # 3. 병렬 코퍼스 구축
    print("\n[3/4] 병렬 코퍼스 구축 중...")
    corpus = build_parallel_corpus(nama_data, eng_data)
    aligned = [e for e in corpus if e["aligned"]]
    print(f"  총 항목: {len(corpus)}개")
    print(f"  정렬 완료: {len(aligned)}개 ({100*len(aligned)/len(corpus):.1f}%)")

    # JSON 저장
    corpus_json = base / "nama_eng_parallel.json"
    with open(corpus_json, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"  → {corpus_json}")

    # CSV 저장
    corpus_csv = base / "nama_eng_parallel.csv"
    with open(corpus_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ref", "book", "chapter", "verse", "nama", "english", "aligned"])
        writer.writeheader()
        writer.writerows(corpus)
    print(f"  → {corpus_csv}")

    # 4. 언어학적 분석
    print("\n[4/4] 나마어 언어학적 분석 중...")
    analysis = analyze_nama_linguistics(corpus)

    analysis_json = base / "linguistics_report.json"
    with open(analysis_json, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    # 리포트 출력
    print(f"\n  총 절수:        {analysis['total_verses']:,}")
    print(f"  총 단어 수:     {analysis['total_words']:,}")
    print(f"  고유 단어 수:   {analysis['unique_words']:,}")
    print(f"  절당 평균 단어: {analysis['avg_words_per_verse']}")
    print(f"  어휘 다양도:    {analysis['type_token_ratio']} (TTR)")
    print(f"  특수 문자:      {analysis['special_characters']}")
    print(f"\n  상위 20개 단어:")
    for word, cnt in analysis["top_50_words"][:20]:
        print(f"    {word:<20} {cnt:>5}회")
    print(f"\n  상위 10개 접미사 패턴:")
    for suf, cnt in analysis["top_20_suffixes"][:10]:
        print(f"    -{suf:<10} {cnt:>5}회")

    print(f"\n  → {analysis_json}")
    print("\n" + "=" * 60)
    print("  완료! 결과 파일:")
    print(f"  코퍼스 JSON: {corpus_json}")
    print(f"  코퍼스 CSV:  {corpus_csv}")
    print(f"  분석 리포트: {analysis_json}")
    print("=" * 60)

    # 샘플 출력
    print("\n  ─── 샘플 병렬 구절 (LUK 1:1~5) ───")
    for entry in corpus[:5]:
        print(f"\n  [{entry['ref']}]")
        print(f"  나마: {entry['nama'][:80]}...")
        print(f"  영어: {entry['english'][:80]}...")


if __name__ == "__main__":
    main()
