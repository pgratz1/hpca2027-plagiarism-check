from pdfsim.textnorm import normalize, split_sentences, tokenize, tokenize_with_offsets


def test_normalize_folds_ligatures_hyphens_and_whitespace():
    raw = "The ﬁrst archi-\ntecture uses “quotes” — and\n  spaces."
    assert normalize(raw) == 'The first architecture uses "quotes" - and spaces.'


def test_hyphen_only_joined_before_lowercase():
    assert normalize("Intel-\nBased") == "Intel- Based"
    assert normalize("low-\npower") == "lowpower"


def test_split_sentences_respects_abbreviations():
    text = "Fig. 3 shows it. See Sec. 2.1 for details. Smith et al. agree! Really? Yes."
    assert split_sentences(text) == [
        "Fig. 3 shows it.", "See Sec. 2.1 for details.", "Smith et al. agree!", "Really?", "Yes."]


def test_split_sentences_min_tokens():
    assert split_sentences("Short. This one has enough tokens in it.", min_tokens=4) == [
        "This one has enough tokens in it."]


def test_tokenize_and_offsets():
    assert tokenize("Low-power CPUs (x86_64) at 3.5GHz.") == ["low", "power", "cpus", "x86", "64", "at", "3", "5ghz"]
    toks = tokenize_with_offsets("Ab cd")
    assert [(t.text, t.start, t.end) for t in toks] == [("ab", 0, 2), ("cd", 3, 5)]
