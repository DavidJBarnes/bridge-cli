"""Tests for scripts/generate_dataset.py -- the Spring Boot dataset generator."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_dataset import (
    CODE_PATTERNS,
    INSTRUCTION_TEMPLATES,
    clean_code,
    clone_github_repo,
    detect_code_type,
    extract_class_name,
    extract_entity_name,
    generate_instruction,
    generate_synthetic_examples,
    main,
    process_java_file,
    scan_directory,
)
from tests.conftest import (
    JAVA_SAMPLES,
    SAMPLE_CONTROLLER_JAVA,
    SAMPLE_CONFIG_JAVA,
    SAMPLE_ENTITY_JAVA,
    SAMPLE_REPOSITORY_JAVA,
    SAMPLE_SERVICE_JAVA,
    SAMPLE_TEST_JAVA,
)


# -----------------------------------------------------------------------
# detect_code_type
# -----------------------------------------------------------------------
class TestDetectCodeType:
    """Tests for detect_code_type covering every pattern and the unknown case."""

    @pytest.mark.parametrize(
        "code_type, sample",
        list(JAVA_SAMPLES.items()),
        ids=list(JAVA_SAMPLES.keys()),
    )
    def test_detects_each_code_pattern(self, code_type, sample):
        """Each CODE_PATTERN type should be detected from its representative sample."""
        assert detect_code_type(sample) == code_type

    def test_returns_none_for_unknown_code(self):
        """Code that matches no pattern should return None."""
        assert detect_code_type("public class PlainPojo { }") is None

    def test_controller_annotation_without_rest_prefix(self):
        """A plain @Controller annotation should also be detected as controller."""
        code = "@Controller\npublic class HomeController {}"
        assert detect_code_type(code) == "controller"

    def test_repository_via_extends_crud_repository(self):
        """Repositories extending CrudRepository should be detected."""
        code = "public interface Repo extends CrudRepository<Foo, Long> {}"
        assert detect_code_type(code) == "repository"


# -----------------------------------------------------------------------
# extract_class_name
# -----------------------------------------------------------------------
class TestExtractClassName:
    """Tests for extract_class_name with various declaration styles."""

    def test_extracts_regular_class(self):
        """Standard public class declaration."""
        assert extract_class_name("public class UserController {}") == "UserController"

    def test_extracts_interface(self):
        """Public interface declaration."""
        assert extract_class_name("public interface ProductRepository {}") == "ProductRepository"

    def test_abstract_class_not_matched(self):
        """The regex expects 'public' immediately followed by 'class' or 'interface'.

        An 'abstract' keyword between them means no match.
        """
        result = extract_class_name("public abstract class BaseService {}")
        assert result is None

    def test_returns_none_when_no_class(self):
        """If there is no class or interface keyword, return None."""
        assert extract_class_name("// just a comment") is None


# -----------------------------------------------------------------------
# extract_entity_name
# -----------------------------------------------------------------------
class TestExtractEntityName:
    """Tests for extract_entity_name -- stripping known suffixes."""

    @pytest.mark.parametrize(
        "class_name, expected",
        [
            ("UserController", "User"),
            ("OrderService", "Order"),
            ("ProductRepository", "Product"),
            ("PaymentImpl", "Payment"),
            ("ApplicationTest", "Application"),
            ("SecurityConfig", "Security"),
            ("LoggingAspect", "Logging"),
            ("GlobalExceptionHandler", "GlobalException"),
            ("OrderListener", "Order"),
            ("EventProducer", "Event"),
            ("NotificationConsumer", "Notification"),
            ("TransactionAdvisor", "Transaction"),
        ],
    )
    def test_strips_known_suffixes(self, class_name, expected):
        """Each known suffix should be removed, yielding the entity root."""
        assert extract_entity_name(class_name) == expected

    def test_returns_original_when_no_suffix(self):
        """A class name with no recognised suffix should be returned as-is."""
        assert extract_entity_name("Invoice") == "Invoice"

    def test_returns_class_name_when_suffix_equals_full_name(self):
        """If removing the suffix leaves an empty string, return original class_name."""
        assert extract_entity_name("Controller") == "Controller"


# -----------------------------------------------------------------------
# generate_instruction
# -----------------------------------------------------------------------
class TestGenerateInstruction:
    """Tests for generate_instruction for every code type."""

    @pytest.mark.parametrize("code_type", list(INSTRUCTION_TEMPLATES.keys()))
    def test_returns_string_for_each_code_type(self, code_type):
        """Every registered code type should produce a non-empty instruction string."""
        instruction = generate_instruction(code_type, "UserController", SAMPLE_CONTROLLER_JAVA)
        assert isinstance(instruction, str)
        assert len(instruction) > 0

    def test_controller_instruction_contains_entity(self):
        """Controller instructions should mention the extracted entity name."""
        with patch("scripts.generate_dataset.random.choice", side_effect=lambda lst: lst[0]):
            result = generate_instruction("controller", "UserController", "")
        assert "User" in result

    def test_config_instruction_contains_purpose(self):
        """Config instructions should format the {purpose} placeholder."""
        with patch("scripts.generate_dataset.random.choice", side_effect=lambda lst: lst[0]):
            result = generate_instruction("config", "SecurityConfig", "")
        assert "Security" in result

    def test_test_instruction_contains_component(self):
        """Test instructions should format the {component} placeholder."""
        with patch("scripts.generate_dataset.random.choice", side_effect=lambda lst: lst[0]):
            result = generate_instruction("test", "ApplicationTest", "")
        assert "ApplicationTest" in result

    def test_unknown_code_type_fallback(self):
        """An unregistered code type should produce the generic fallback string."""
        result = generate_instruction("unknown_type", "FooBar", "")
        assert "FooBar" in result
        assert "unknown_type" in result


# -----------------------------------------------------------------------
# clean_code
# -----------------------------------------------------------------------
class TestCleanCode:
    """Tests for clean_code -- package, import, and blank-line removal."""

    def test_removes_package_declaration(self):
        """Package lines should be stripped."""
        code = "package com.example.demo;\n\npublic class Foo {}"
        assert "package" not in clean_code(code)

    def test_removes_import_statements(self):
        """All import lines should be removed."""
        code = "import java.util.List;\nimport java.util.Map;\npublic class Foo {}"
        cleaned = clean_code(code)
        assert "import" not in cleaned

    def test_collapses_excessive_blank_lines(self):
        """Three or more consecutive blank lines should be collapsed to two."""
        code = "class Foo {\n\n\n\n\n    int x;\n}"
        cleaned = clean_code(code)
        assert "\n\n\n" not in cleaned

    def test_strips_leading_and_trailing_whitespace(self):
        """Output should have no leading/trailing whitespace."""
        code = "\n\n  public class Foo {}  \n\n"
        cleaned = clean_code(code)
        assert cleaned == cleaned.strip()


# -----------------------------------------------------------------------
# process_java_file
# -----------------------------------------------------------------------
class TestProcessJavaFile:
    """Tests for process_java_file with various file scenarios."""

    def _write_java(self, tmp_path, filename, content):
        """Helper to write a Java file and return its Path."""
        p = tmp_path / filename
        p.write_text(content, encoding="utf-8")
        return p

    def test_valid_controller_file(self, tmp_path):
        """A well-formed controller file should produce a training example dict."""
        content = SAMPLE_CONTROLLER_JAVA * 5
        path = self._write_java(tmp_path, "UserController.java", content)
        result = process_java_file(path)
        assert result is not None
        assert "instruction" in result
        assert "input" in result
        assert "output" in result
        assert result["input"] == ""

    def test_file_too_short(self, tmp_path):
        """Files whose cleaned output is < min_examples should be skipped."""
        short = "@Service\npublic class Tiny { }"
        path = self._write_java(tmp_path, "Tiny.java", short)
        assert process_java_file(path) is None

    def test_file_too_long(self, tmp_path):
        """Files whose cleaned output exceeds max_length should be skipped."""
        long_content = "@Service\npublic class Big {\n" + ("    int x;\n" * 5000) + "}\n"
        path = self._write_java(tmp_path, "Big.java", long_content)
        assert process_java_file(path) is None

    def test_custom_min_max_params(self, tmp_path):
        """Custom min_examples and max_length should be respected."""
        content = "@Service\npublic class Svc {\n" + ("    int a;\n" * 10) + "}\n"
        path = self._write_java(tmp_path, "Svc.java", content)
        # With a very low min and high max, it should succeed
        result = process_java_file(path, min_examples=10, max_length=50000)
        assert result is not None

    def test_unreadable_file(self, tmp_path):
        """Files that cannot be read should return None without raising."""
        path = tmp_path / "Ghost.java"
        path.write_text("content", encoding="utf-8")
        path.chmod(0o000)
        result = process_java_file(path)
        # Restore permissions for cleanup
        path.chmod(0o644)
        assert result is None

    def test_no_detected_type(self, tmp_path):
        """Files with no recognisable Spring Boot annotation should be skipped."""
        plain = "public class PlainPojo { private int id; }"
        path = self._write_java(tmp_path, "PlainPojo.java", plain)
        assert process_java_file(path) is None

    def test_test_file_is_skipped(self, tmp_path):
        """Files detected as code_type 'test' should be skipped."""
        content = SAMPLE_TEST_JAVA * 10
        path = self._write_java(tmp_path, "ApplicationTest.java", content)
        assert process_java_file(path) is None

    def test_no_class_name_returns_none(self, tmp_path):
        """Files where extract_class_name returns None should be skipped."""
        content = "@Service\nclass InternalService { }" + "\n// filler\n" * 200
        path = self._write_java(tmp_path, "Internal.java", content)
        assert process_java_file(path) is None


# -----------------------------------------------------------------------
# scan_directory
# -----------------------------------------------------------------------
class TestScanDirectory:
    """Tests for scan_directory with a temp directory containing Java files."""

    def test_finds_examples_in_main_source(self, java_temp_dir):
        """Java files under src/main should be processed; files under /test/ should be skipped."""
        examples = scan_directory(java_temp_dir)
        assert isinstance(examples, list)
        assert len(examples) >= 1

    def test_skips_test_directory(self, java_temp_dir):
        """Files whose path contains '/test/' should not appear in results."""
        examples = scan_directory(java_temp_dir)
        for ex in examples:
            assert "ApplicationTest" not in ex.get("instruction", "")

    def test_empty_directory(self, tmp_path):
        """An empty directory should return an empty list."""
        assert scan_directory(tmp_path) == []

    def test_forwards_min_max_params(self, java_temp_dir):
        """min_examples and max_length params should be forwarded to process_java_file."""
        # With an extremely high min, nothing should pass
        examples = scan_directory(java_temp_dir, min_examples=999999)
        assert examples == []

    def test_progress_logging(self, tmp_path):
        """When many files are processed, the progress logger at every 50 files should fire.

        We create 51 valid Java files to trigger the progress log on the 50th.
        """
        src_dir = tmp_path / "src" / "main" / "java"
        src_dir.mkdir(parents=True)
        # Use a short min_examples so all files pass
        for i in range(51):
            content = f"@Service\npublic class Svc{i} {{\n" + ("    int a;\n" * 30) + "}\n"
            (src_dir / f"Svc{i}.java").write_text(content, encoding="utf-8")

        examples = scan_directory(tmp_path, min_examples=10)
        assert len(examples) == 51


# -----------------------------------------------------------------------
# clone_github_repo
# -----------------------------------------------------------------------
class TestCloneGithubRepo:
    """Tests for clone_github_repo with mocked subprocess."""

    def test_successful_clone(self, tmp_path, mock_subprocess_run):
        """A successful git clone should return True."""
        target = tmp_path / "new-repo"
        assert clone_github_repo("owner/repo", target) is True
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args
        assert "git" in call_args[0][0]

    def test_clone_failure(self, tmp_path, mock_subprocess_run):
        """A CalledProcessError from git should cause the function to return False."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "git")
        target = tmp_path / "fail-repo"
        assert clone_github_repo("owner/repo", target) is False

    def test_skips_when_directory_exists(self, tmp_path, mock_subprocess_run):
        """If the target directory already exists, clone is skipped and True returned."""
        existing = tmp_path / "existing-repo"
        existing.mkdir()
        assert clone_github_repo("owner/repo", existing) is True
        mock_subprocess_run.assert_not_called()


# -----------------------------------------------------------------------
# generate_synthetic_examples
# -----------------------------------------------------------------------
class TestGenerateSyntheticExamples:
    """Tests for generate_synthetic_examples."""

    def test_returns_non_empty_list(self):
        """The function should always return at least one example."""
        examples = generate_synthetic_examples()
        assert len(examples) > 0

    def test_examples_have_correct_keys(self):
        """Every example must have instruction, input, and output keys."""
        for ex in generate_synthetic_examples():
            assert "instruction" in ex
            assert "input" in ex
            assert "output" in ex


# -----------------------------------------------------------------------
# main (CLI argument parsing & orchestration)
# -----------------------------------------------------------------------
class TestMain:
    """Tests for the main function's argument parsing and execution flow."""

    def test_default_arguments(self, tmp_path, mocker):
        """With minimal arguments, main should process default repos and write output."""
        output_file = tmp_path / "output.jsonl"
        mocker.patch(
            "sys.argv",
            ["generate_dataset.py", "--output", str(output_file)],
        )
        mocker.patch("scripts.generate_dataset.clone_github_repo", return_value=False)

        main()

        assert output_file.exists()

    def test_with_local_dir(self, java_temp_dir, tmp_path, mocker):
        """The --local-dir flag should scan the given directory."""
        output_file = tmp_path / "out.jsonl"
        mocker.patch(
            "sys.argv",
            [
                "generate_dataset.py",
                "--output", str(output_file),
                "--local-dir", str(java_temp_dir),
            ],
        )
        mocker.patch("scripts.generate_dataset.clone_github_repo", return_value=False)

        main()

        assert output_file.exists()
        lines = output_file.read_text().strip().splitlines()
        assert len(lines) >= 1
        for line in lines:
            data = json.loads(line)
            assert "instruction" in data

    def test_with_nonexistent_local_dir(self, tmp_path, mocker):
        """When --local-dir points to a non-existent path, the error branch should execute."""
        output_file = tmp_path / "out.jsonl"
        mocker.patch(
            "sys.argv",
            [
                "generate_dataset.py",
                "--output", str(output_file),
                "--local-dir", "/nonexistent/path/xyz",
            ],
        )
        mocker.patch("scripts.generate_dataset.clone_github_repo", return_value=False)

        main()

        # The output file should still be created (empty)
        assert output_file.exists()

    def test_with_synthetic_flag(self, tmp_path, mocker):
        """The --add-synthetic flag should append synthetic examples."""
        output_file = tmp_path / "synth.jsonl"
        mocker.patch(
            "sys.argv",
            [
                "generate_dataset.py",
                "--output", str(output_file),
                "--add-synthetic",
            ],
        )
        mocker.patch("scripts.generate_dataset.clone_github_repo", return_value=False)

        main()

        lines = output_file.read_text().strip().splitlines()
        assert len(lines) >= len(generate_synthetic_examples())

    def test_with_custom_github_repo(self, tmp_path, mocker):
        """The --github-repo flag should override the default repo list."""
        output_file = tmp_path / "custom.jsonl"
        mocker.patch(
            "sys.argv",
            [
                "generate_dataset.py",
                "--output", str(output_file),
                "--github-repo", "custom/repo",
            ],
        )
        mock_clone = mocker.patch(
            "scripts.generate_dataset.clone_github_repo", return_value=False
        )

        main()

        assert mock_clone.call_count == 1
        assert "custom/repo" in mock_clone.call_args[0][0]

    def test_successful_clone_triggers_scan(self, tmp_path, mocker):
        """When clone succeeds, scan_directory should be called for that repo."""
        output_file = tmp_path / "scanned.jsonl"
        mocker.patch(
            "sys.argv",
            [
                "generate_dataset.py",
                "--output", str(output_file),
                "--github-repo", "owner/myrepo",
            ],
        )
        mocker.patch("scripts.generate_dataset.clone_github_repo", return_value=True)
        mock_scan = mocker.patch(
            "scripts.generate_dataset.scan_directory", return_value=[]
        )

        main()

        mock_scan.assert_called_once()

    def test_min_examples_and_max_length_args(self, tmp_path, mocker):
        """--min-examples and --max-length should be forwarded to scan_directory."""
        output_file = tmp_path / "params.jsonl"
        mocker.patch(
            "sys.argv",
            [
                "generate_dataset.py",
                "--output", str(output_file),
                "--github-repo", "owner/repo",
                "--min-examples", "500",
                "--max-length", "8000",
            ],
        )
        mocker.patch("scripts.generate_dataset.clone_github_repo", return_value=True)
        mock_scan = mocker.patch(
            "scripts.generate_dataset.scan_directory", return_value=[]
        )

        main()

        _, kwargs = mock_scan.call_args
        assert kwargs["min_examples"] == 500
        assert kwargs["max_length"] == 8000

    def test_output_write_failure(self, tmp_path, mocker):
        """When writing the output file fails with OSError, SystemExit(1) should be raised."""
        # Use a path under a non-writable directory
        locked_dir = tmp_path / "locked"
        locked_dir.mkdir()
        output_file = locked_dir / "sub" / "out.jsonl"

        mocker.patch(
            "sys.argv",
            [
                "generate_dataset.py",
                "--output", str(output_file),
                "--add-synthetic",
            ],
        )
        mocker.patch("scripts.generate_dataset.clone_github_repo", return_value=False)
        # Make the parent mkdir succeed but open() fail
        mocker.patch("builtins.open", side_effect=OSError("disk full"))

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
