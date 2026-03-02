"""Tests for the custom parser plugin system (addresses #151)."""

import pytest

from raganything.parser import (
    Parser,
    get_parser,
    register_parser,
    unregister_parser,
    list_parsers,
    get_supported_parsers,
    _CUSTOM_PARSERS,
    SUPPORTED_PARSERS,
)


class DummyParser(Parser):
    """Minimal custom parser for testing."""

    def check_installation(self) -> bool:
        return True

    def parse_document(self, file_path, output_dir="./output", method="auto", **kw):
        return [{"type": "text", "text": "dummy parsed content", "page_idx": 0}]

    def parse_pdf(self, pdf_path, output_dir="./output", method="auto", **kw):
        return self.parse_document(file_path=pdf_path, output_dir=output_dir, method=method, **kw)


class AnotherParser(Parser):
    """Another custom parser for testing."""

    def check_installation(self) -> bool:
        return False

    def parse_document(self, file_path, output_dir="./output", method="auto", **kw):
        return [{"type": "text", "text": "another parsed", "page_idx": 0}]


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure a clean custom parser registry for every test."""
    _CUSTOM_PARSERS.clear()
    yield
    _CUSTOM_PARSERS.clear()


class TestRegisterParser:
    def test_register_and_get(self):
        register_parser("dummy", DummyParser)
        parser = get_parser("dummy")
        assert isinstance(parser, DummyParser)

    def test_register_case_insensitive(self):
        register_parser("  Dummy  ", DummyParser)
        parser = get_parser("dummy")
        assert isinstance(parser, DummyParser)

    def test_register_rejects_non_parser_subclass(self):
        with pytest.raises(TypeError, match="subclass of Parser"):
            register_parser("bad", dict)

    def test_register_rejects_builtin_name(self):
        for name in ("mineru", "docling", "paddleocr"):
            with pytest.raises(ValueError, match="Cannot override built-in"):
                register_parser(name, DummyParser)

    def test_register_overwrites_same_custom_name(self):
        register_parser("custom", DummyParser)
        register_parser("custom", AnotherParser)
        parser = get_parser("custom")
        assert isinstance(parser, AnotherParser)


class TestUnregisterParser:
    def test_unregister_existing(self):
        register_parser("dummy", DummyParser)
        unregister_parser("dummy")
        with pytest.raises(ValueError, match="Unsupported parser type"):
            get_parser("dummy")

    def test_unregister_nonexistent(self):
        with pytest.raises(KeyError, match="No custom parser"):
            unregister_parser("nonexistent")


class TestListParsers:
    def test_list_builtin_only(self):
        result = list_parsers()
        assert "mineru" in result
        assert "docling" in result
        assert "paddleocr" in result
        assert len(result) == 3

    def test_list_includes_custom(self):
        register_parser("dummy", DummyParser)
        result = list_parsers()
        assert "dummy" in result
        assert result["dummy"] == "DummyParser"
        assert len(result) == 4


class TestGetSupportedParsers:
    def test_builtin_only(self):
        supported = get_supported_parsers()
        assert set(SUPPORTED_PARSERS).issubset(set(supported))

    def test_includes_custom(self):
        register_parser("dummy", DummyParser)
        supported = get_supported_parsers()
        assert "dummy" in supported


class TestGetParserFallback:
    def test_builtin_parsers_still_work(self):
        """Ensure built-in parsers are unaffected by the plugin system."""
        for name in SUPPORTED_PARSERS:
            parser = get_parser(name)
            assert isinstance(parser, Parser)

    def test_unknown_parser_raises(self):
        with pytest.raises(ValueError, match="Unsupported parser type"):
            get_parser("totally-unknown")

    def test_custom_parser_content(self):
        register_parser("dummy", DummyParser)
        parser = get_parser("dummy")
        content = parser.parse_document("fake.pdf")
        assert content == [{"type": "text", "text": "dummy parsed content", "page_idx": 0}]
