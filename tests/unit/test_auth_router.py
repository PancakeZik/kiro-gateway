# -*- coding: utf-8 -*-

"""Tests for multi-account auth routing."""

from unittest.mock import MagicMock

import pytest

from kiro.auth_router import is_haiku_model, resolve_auth_manager


class TestIsHaikuModel:
    """Tests for haiku model detection."""

    def test_haiku_exact(self):
        assert is_haiku_model("claude-haiku-4.5") is True

    def test_haiku_case_insensitive(self):
        assert is_haiku_model("Claude-Haiku-4.5") is True
        assert is_haiku_model("CLAUDE-HAIKU-4.5") is True

    def test_haiku_substring(self):
        assert is_haiku_model("claude-haiku-4") is True
        assert is_haiku_model("some-haiku-model") is True

    def test_opus_not_haiku(self):
        assert is_haiku_model("claude-opus-4.5") is False

    def test_sonnet_not_haiku(self):
        assert is_haiku_model("claude-sonnet-4") is False
        assert is_haiku_model("claude-sonnet-4.5") is False

    def test_auto_not_haiku(self):
        assert is_haiku_model("auto") is False
        assert is_haiku_model("auto-kiro") is False

    def test_legacy_sonnet_not_haiku(self):
        assert is_haiku_model("claude-3.7-sonnet") is False


class TestResolveAuthManager:
    """Tests for auth manager resolution based on model."""

    def setup_method(self):
        self.primary_auth = MagicMock(name="primary_auth")
        self.haiku_auth = MagicMock(name="haiku_auth")

    def test_haiku_model_returns_haiku_auth(self):
        result = resolve_auth_manager("claude-haiku-4.5", self.primary_auth, self.haiku_auth)
        assert result is self.haiku_auth

    def test_opus_model_returns_primary_auth(self):
        result = resolve_auth_manager("claude-opus-4.5", self.primary_auth, self.haiku_auth)
        assert result is self.primary_auth

    def test_sonnet_model_returns_primary_auth(self):
        result = resolve_auth_manager("claude-sonnet-4", self.primary_auth, self.haiku_auth)
        assert result is self.primary_auth

    def test_auto_returns_primary_auth(self):
        result = resolve_auth_manager("auto", self.primary_auth, self.haiku_auth)
        assert result is self.primary_auth

    def test_unknown_model_returns_primary_auth(self):
        result = resolve_auth_manager("some-future-model", self.primary_auth, self.haiku_auth)
        assert result is self.primary_auth
