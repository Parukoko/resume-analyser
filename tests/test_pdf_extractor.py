from unittest.mock import patch

from PIL import Image

from app.extractors import pdf_extractor


def test_text_based_pdf_uses_pdfplumber_path(sample_resume_pdf):
    result = pdf_extractor.extract_text(str(sample_resume_pdf))
    assert result.method == "text"
    assert "Jane Doe" in result.text


def test_looks_like_scanned_thresholds():
    assert pdf_extractor._looks_like_scanned("x", page_count=1) is True
    assert pdf_extractor._looks_like_scanned("a" * 500, page_count=1) is False
    assert pdf_extractor._looks_like_scanned("", page_count=0) is False


def test_page_to_data_url_is_a_valid_png_data_url():
    img = Image.new("RGB", (10, 10), color="white")
    url = pdf_extractor._page_to_data_url(img)
    assert url.startswith("data:image/png;base64,")


def _fake_openai_response(content: str):
    msg = type("M", (), {"content": content})()
    choice = type("C", (), {"message": msg})()
    return type("R", (), {"choices": [choice]})()


def test_transcribe_page_sends_openai_vision_message_shape():
    img = Image.new("RGB", (10, 10), color="white")

    with patch.object(
        pdf_extractor._client.chat.completions, "create", side_effect=lambda **kw: _fake_openai_response("hi")
    ) as mock_create:
        text = pdf_extractor._transcribe_page(img)

    assert text == "hi"
    content = mock_create.call_args.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_page_transcription_recovers_after_one_transient_failure():
    img = Image.new("RGB", (10, 10), color="white")
    calls = {"n": 0}

    def fails_once_then_succeeds(image):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient failure")
        return "recovered"

    with patch.object(pdf_extractor, "_transcribe_page", side_effect=fails_once_then_succeeds), patch.object(
        pdf_extractor.time, "sleep"
    ) as mock_sleep:
        text = pdf_extractor._transcribe_page_with_retry(img, page_num=1)

    assert text == "recovered"
    assert calls["n"] == 2
    mock_sleep.assert_called_once_with(pdf_extractor.PAGE_TRANSCRIPTION_RETRY_DELAY_SECONDS)


def test_vision_extraction_survives_a_page_that_fails_even_after_retry():
    img = Image.new("RGB", (10, 10), color="white")
    calls = {"n": 0}

    def flaky(image):
        calls["n"] += 1
        # page 1: succeeds first try (call 1). page 2: fails both the try and
        # the retry (calls 2 and 3). page 3: succeeds first try (call 4).
        if calls["n"] in (2, 3):
            raise RuntimeError("simulated failure")
        return f"call {calls['n']}"

    with patch("pdf2image.convert_from_path", return_value=[img, img, img]), patch.object(
        pdf_extractor, "_transcribe_page", side_effect=flaky
    ), patch.object(pdf_extractor.time, "sleep"):
        text = pdf_extractor._extract_with_vision_llm("/fake.pdf")

    assert "call 1" in text
    assert "[page 2: transcription failed]" in text
    assert "call 4" in text


def test_extract_text_falls_back_to_vision_for_sparse_pdf(tmp_path):
    # _extract_with_pdfplumber is mocked below, so this only needs to exist as a path.
    sparse_pdf = tmp_path / "sparse.pdf"
    sparse_pdf.write_bytes(b"%PDF-1.4 placeholder")

    img = Image.new("RGB", (10, 10), color="white")
    long_transcription = "Transcribed text " * 20

    with patch("app.extractors.pdf_extractor._extract_with_pdfplumber", return_value=("x", 1)), patch(
        "pdf2image.convert_from_path", return_value=[img]
    ), patch.object(
        pdf_extractor._client.chat.completions,
        "create",
        side_effect=lambda **kw: _fake_openai_response(long_transcription),
    ):
        result = pdf_extractor.extract_text(str(sparse_pdf))

    assert result.method == "vision"
    assert "Transcribed text" in result.text
