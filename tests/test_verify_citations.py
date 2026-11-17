"""
The fabricated-citation guard is only trustworthy if the DOIs it is handed are
looked up in the form Crossref expects. A DOI copied out of a reference list
usually arrives as a doi.org URL, not the bare form -- if that URL form is sent
to the API unprocessed it 404s, and a real citation gets branded "likely
fabricated". These tests pin the normalization, the URL encoding, and the
title-match rule (a claimed title that is merely a subset of the real one must
not score a perfect match -- the same failure mode dedup.py rejects
token_set_ratio for).
"""
import verify_citations as vc


def _mock_lookup(monkeypatch, real_title, authors="Smith, J"):
    captured = {}

    def fake(doi):
        captured["doi"] = doi
        return True, real_title, authors

    monkeypatch.setattr(vc, "crossref_lookup", fake)
    return captured


class _Resp:
    status_code = 200

    def json(self):
        return {"message": {"title": ["T"], "author": [{"family": "Smith"}]}}


class TestDoiNormalization:
    def test_url_form_doi_is_normalized_before_lookup(self, monkeypatch):
        captured = _mock_lookup(monkeypatch, "Some Real Title")
        vc.verify_one("https://doi.org/10.1002/ett.4150", "Some Real Title")
        assert captured["doi"] == "10.1002/ett.4150"

    def test_doi_prefix_form_is_normalized(self, monkeypatch):
        captured = _mock_lookup(monkeypatch, "Some Real Title")
        vc.verify_one("doi:10.1002/ett.4150", "Some Real Title")
        assert captured["doi"] == "10.1002/ett.4150"


class TestTitleMatch:
    def test_subset_title_does_not_pass(self, monkeypatch):
        _mock_lookup(
            monkeypatch,
            "Machine Learning for Intrusion Detection: A Systematic Study "
            "of Deep Learning Approaches",
        )
        r = vc.verify_one("10.1000/xyz", "Machine Learning for Intrusion Detection")
        assert r["status"] == "FAIL"
        assert r["title_score"] < vc.TITLE_MATCH_THRESHOLD

    def test_exact_title_passes(self, monkeypatch):
        _mock_lookup(monkeypatch, "Machine Learning for Intrusion Detection")
        r = vc.verify_one("10.1000/xyz", "Machine Learning for Intrusion Detection")
        assert r["status"] == "PASS"
        assert r["title_score"] == 100


class TestCrossrefRequest:
    def test_doi_is_url_encoded(self, monkeypatch):
        seen = {}

        def fake_get(url, **kw):
            seen["url"] = url
            seen.update(kw)
            return _Resp()

        monkeypatch.setattr(vc.requests, "get", fake_get)
        vc.crossref_lookup("10.1000/a b#c")
        assert "a%20b%23c" in seen["url"]

    def test_mailto_user_agent_sent_when_configured(self, monkeypatch):
        seen = {}

        def fake_get(url, **kw):
            seen.update(kw)
            return _Resp()

        monkeypatch.setattr(vc.requests, "get", fake_get)
        monkeypatch.setenv("MAILTO", "reviewer@example.org")
        vc.crossref_lookup("10.1000/xyz")
        ua = (seen.get("headers") or {}).get("User-Agent", "")
        assert "reviewer@example.org" in ua
