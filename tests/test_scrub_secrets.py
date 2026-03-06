"""Tests for the secret scrubbing utility.

Covers pattern matching and file scrubbing for all secret types
detected by the scrub_secrets module.
"""

import json

import pytest

from scripts.scrub_secrets import COMBINED, scrub_file


class TestCombinedPattern:
    """Tests for the compiled secret detection regex."""

    def test_matches_gocspx_token(self):
        """Verify Google OAuth client secret pattern is detected."""
        assert COMBINED.search("GOCSPX-abc123_DEF-456")

    def test_matches_client_secret(self):
        """Verify generic client_secret keyword is detected."""
        assert COMBINED.search('"client_secret": "something"')

    def test_matches_google_client(self):
        """Verify GOOGLE_CLIENT keyword is detected."""
        assert COMBINED.search("GOOGLE_CLIENT_ID=123456")

    def test_matches_google_client_secret(self):
        """Verify GOOGLE_CLIENT_SECRET keyword is detected."""
        assert COMBINED.search("GOOGLE_CLIENT_SECRET=abcdef")

    def test_matches_airtable_pat(self):
        """Verify Airtable personal access token pattern is detected."""
        assert COMBINED.search("airtable_key=patAbCdEfGhIjKlMn.something")

    def test_matches_airtable_api_key(self):
        """Verify AIRTABLE_API_KEY keyword is detected."""
        assert COMBINED.search("AIRTABLE_API_KEY=keyAbcdef")

    def test_matches_google_oauth_app_id(self):
        """Verify Google OAuth app ID pattern is detected."""
        assert COMBINED.search("123456789-abcdef.apps.googleusercontent.com")

    def test_matches_google_api_key(self):
        """Verify Google API key (AIza) pattern is detected."""
        assert COMBINED.search("AIzaSyB_abcdefghijklmnopqrstuvwxyz12345")

    def test_matches_openai_key(self):
        """Verify OpenAI-style sk- key pattern is detected."""
        assert COMBINED.search("sk-abcdefghijklmnopqrstuvwxyz")

    def test_matches_github_pat(self):
        """Verify GitHub personal access token pattern is detected."""
        assert COMBINED.search("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij")

    def test_matches_private_key(self):
        """Verify RSA private key header is detected."""
        assert COMBINED.search("-----BEGIN RSA PRIVATE KEY-----")

    def test_matches_ec_private_key(self):
        """Verify EC private key header is detected."""
        assert COMBINED.search("-----BEGIN EC PRIVATE KEY-----")

    def test_matches_generic_private_key(self):
        """Verify generic private key header is detected."""
        assert COMBINED.search("-----BEGIN PRIVATE KEY-----")

    def test_no_match_on_clean_code(self):
        """Verify clean code lines are not flagged."""
        assert not COMBINED.search('const api = axios.create({ baseURL: "/api" })')

    def test_no_match_on_normal_string(self):
        """Verify normal strings are not flagged."""
        assert not COMBINED.search("Hello world, this is a test")

    def test_case_insensitive(self):
        """Verify pattern matching is case insensitive."""
        assert COMBINED.search("google_client_id=test")
        assert COMBINED.search("Client_Secret=test")


class TestScrubFile:
    """Tests for the file scrubbing function."""

    def test_removes_lines_with_secrets(self, tmp_path):
        """Verify lines containing secrets are removed."""
        filepath = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"instruction": "clean example", "output": "hello"}) + "\n",
            json.dumps({"instruction": "has secret", "output": "GOCSPX-abc123"}) + "\n",
            json.dumps({"instruction": "also clean", "output": "world"}) + "\n",
        ]
        filepath.write_text("".join(lines))

        scrub_file(str(filepath))

        result = filepath.read_text().strip().split("\n")
        assert len(result) == 2
        assert "clean example" in result[0]
        assert "also clean" in result[1]

    def test_preserves_clean_file(self, tmp_path):
        """Verify a file with no secrets is unchanged."""
        filepath = tmp_path / "clean.jsonl"
        lines = [
            json.dumps({"instruction": "test1", "output": "code"}) + "\n",
            json.dumps({"instruction": "test2", "output": "more code"}) + "\n",
        ]
        filepath.write_text("".join(lines))

        scrub_file(str(filepath))

        result = filepath.read_text().strip().split("\n")
        assert len(result) == 2

    def test_removes_multiple_secret_types(self, tmp_path):
        """Verify multiple different secret types are all removed."""
        filepath = tmp_path / "multi.jsonl"
        lines = [
            "clean line\n",
            "has GOOGLE_CLIENT_ID=123\n",
            "has client_secret in it\n",
            "has ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij\n",
            "another clean line\n",
        ]
        filepath.write_text("".join(lines))

        scrub_file(str(filepath))

        result = filepath.read_text().strip().split("\n")
        assert len(result) == 2
        assert "clean line" in result[0]
        assert "another clean line" in result[1]

    def test_empty_file(self, tmp_path):
        """Verify empty files are handled gracefully."""
        filepath = tmp_path / "empty.jsonl"
        filepath.write_text("")

        scrub_file(str(filepath))

        assert filepath.read_text() == ""

    def test_all_lines_removed(self, tmp_path):
        """Verify file is empty when all lines contain secrets."""
        filepath = tmp_path / "allsecrets.jsonl"
        lines = [
            "GOOGLE_CLIENT_SECRET=abc\n",
            "sk-abcdefghijklmnopqrstuvwxyz\n",
        ]
        filepath.write_text("".join(lines))

        scrub_file(str(filepath))

        assert filepath.read_text() == ""
