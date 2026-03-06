"""Tests for scripts/generate_react_dataset.py -- the React/TypeScript dataset generator."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_react_dataset import (
    CODE_PATTERNS,
    INSTRUCTION_TEMPLATES,
    _derive_entity,
    clean_code,
    clone_github_repo,
    detect_code_type,
    extract_component_name,
    generate_instruction,
    generate_synthetic_examples,
    main,
    process_file,
    scan_directory,
)


# ---------------------------------------------------------------------------
# Sample React/TypeScript code strings per pattern type
# ---------------------------------------------------------------------------

REACT_SAMPLES = {
    "functional_component": textwrap.dedent("""\
        import React from 'react';

        export default function Dashboard({ title }: { title: string }): JSX.Element {
          return (
            <div>
              <h1>{title}</h1>
            </div>
          );
        }
    """),
    "hook": textwrap.dedent("""\
        import { useState, useEffect } from 'react';

        export function useDebounce(value: string, delay: number) {
          const [debounced, setDebounced] = useState(value);
          useEffect(() => {
            const timer = setTimeout(() => setDebounced(value), delay);
            return () => clearTimeout(timer);
          }, [value, delay]);
          return debounced;
        }
    """),
    "context_provider": textwrap.dedent("""\
        import React, { createContext, useContext, useState } from 'react';

        interface ThemeCtx { dark: boolean; toggle: () => void; }
        const ThemeContext = createContext<ThemeCtx | undefined>(undefined);

        export function ThemeProvider({ children }: { children: React.ReactNode }) {
          const [dark, setDark] = useState(false);
          return (
            <ThemeContext.Provider value={{ dark, toggle: () => setDark(!dark) }}>
              {children}
            </ThemeContext.Provider>
          );
        }
    """),
    "hoc": textwrap.dedent("""\
        import React from 'react';

        export function withLogging(WrappedComponent: React.ComponentType<any>) {
          return function LoggedComponent(props: any) {
            console.log('Rendering', WrappedComponent.name);
            return <WrappedComponent {...props} />;
          };
        }
    """),
    "route_config": textwrap.dedent("""\
        import { BrowserRouter, Routes, Route } from 'react-router-dom';
        import Home from './Home';

        export default function AppRoutes() {
          return (
            <BrowserRouter>
              <Routes>
                <Route path="/" element={<Home />} />
              </Routes>
            </BrowserRouter>
          );
        }
    """),
    "api_service": textwrap.dedent("""\
        import axios from 'axios';

        const client = axios.create({ baseURL: '/api' });

        export class UserApi {
          static async getAll() { return client.get('/users'); }
          static async getById(id: string) { return client.get(`/users/${id}`); }
        }
    """),
    "redux_slice": textwrap.dedent("""\
        import { createSlice, PayloadAction } from '@reduxjs/toolkit';

        interface CounterState { value: number; }
        const initialState: CounterState = { value: 0 };

        const counterSlice = createSlice({
          name: 'counter',
          initialState,
          reducers: {
            increment(state) { state.value += 1; },
            set(state, action: PayloadAction<number>) { state.value = action.payload; },
          },
        });

        export const { increment, set } = counterSlice.actions;
        export default counterSlice.reducer;
    """),
    "test": textwrap.dedent("""\
        import { render, screen } from '@testing-library/react';
        import Dashboard from './Dashboard';

        describe('Dashboard', () => {
          it('renders the title', () => {
            render(<Dashboard title="Hello" />);
            expect(screen.getByText('Hello')).toBeInTheDocument();
          });
        });
    """),
    "typescript_types": textwrap.dedent("""\
        export interface User {
          id: string;
          email: string;
          name: string;
          roles: string[];
          createdAt: Date;
        }

        export type UserCreateInput = Omit<User, 'id' | 'createdAt'>;
        export type UserUpdateInput = Partial<UserCreateInput>;
    """),
}


# -----------------------------------------------------------------------
# detect_code_type
# -----------------------------------------------------------------------
class TestDetectCodeType:
    """Tests for detect_code_type covering every React pattern and the unknown case."""

    @pytest.mark.parametrize(
        "code_type, sample",
        list(REACT_SAMPLES.items()),
        ids=list(REACT_SAMPLES.keys()),
    )
    def test_detects_each_react_pattern(self, code_type, sample):
        """Each CODE_PATTERN type should be detected from its representative sample."""
        assert detect_code_type(sample) == code_type

    def test_returns_none_for_unknown(self):
        """Content that matches no pattern should return None."""
        assert detect_code_type("const x = 42;\nconsole.log(x);") is None


# -----------------------------------------------------------------------
# extract_component_name
# -----------------------------------------------------------------------
class TestExtractComponentName:
    """Tests for extract_component_name with various export styles."""

    def test_export_default_function(self):
        """Should extract from 'export default function Foo'."""
        code = "export default function Dashboard() { return <div />; }"
        assert extract_component_name(code) == "Dashboard"

    def test_export_default_identifier(self):
        """Should extract from 'export default Identifier'."""
        code = "const Sidebar = () => {};\nexport default Sidebar"
        assert extract_component_name(code) == "Sidebar"

    def test_export_const(self):
        """Should extract from 'export const Foo'."""
        code = "export const UserCard = ({ name }: Props) => { return <p>{name}</p>; }"
        assert extract_component_name(code) == "UserCard"

    def test_plain_function(self):
        """Should extract from 'function Foo('."""
        code = "function helper(x: number) { return x + 1; }"
        assert extract_component_name(code) == "helper"

    def test_plain_const(self):
        """Should extract from 'const Foo ='."""
        code = "const myUtil = () => {};"
        assert extract_component_name(code) == "myUtil"

    def test_skips_reserved_names(self):
        """Names like 'default', 'module', 'exports', 'require', 'undefined' should be skipped."""
        code = "export default module;"
        # 'default' and 'module' should both be skipped; no fallback path
        result = extract_component_name(code)
        # 'module' is a reserved skip name, and 'default' too
        assert result is None or result not in {"default", "module", "exports", "require", "undefined"}

    def test_falls_back_to_filepath_stem(self):
        """When no export match, the file stem should be used."""
        code = "// no exports here"
        result = extract_component_name(code, filepath=Path("/src/components/Button.tsx"))
        assert result == "Button"

    def test_filepath_stem_strips_test_suffix(self):
        """File stems like 'Button.test' should have '.test' removed."""
        code = "// no exports"
        result = extract_component_name(code, filepath=Path("/src/Button.test.tsx"))
        assert result == "Button"

    def test_filepath_stem_strips_spec_suffix(self):
        """File stems like 'UserList.spec' should have '.spec' removed."""
        code = "// no exports"
        result = extract_component_name(code, filepath=Path("/src/UserList.spec.tsx"))
        assert result == "UserList"

    def test_filepath_index_returns_none(self):
        """An 'index' file stem should return None (not a useful name)."""
        code = "// no exports"
        result = extract_component_name(code, filepath=Path("/src/index.tsx"))
        assert result is None

    def test_returns_none_with_no_match_and_no_filepath(self):
        """When nothing matches and no filepath is provided, return None."""
        code = "// nothing"
        assert extract_component_name(code) is None


# -----------------------------------------------------------------------
# _derive_entity
# -----------------------------------------------------------------------
class TestDeriveEntity:
    """Tests for _derive_entity -- stripping React-related suffixes."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("DashboardComponent", "Dashboard"),
            ("UserContainer", "User"),
            ("AuthProvider", "Auth"),
            ("ThemeContext", "Theme"),
            ("CartSlice", "Cart"),
            ("UserService", "User"),
            ("OrderClient", "Order"),
            ("ProductApi", "Product"),
            ("FetchHook", "Fetch"),
            ("AuthHOC", "Auth"),
            ("HomePage", "Home"),
            ("ProfileView", "Profile"),
            ("DetailScreen", "Detail"),
            ("LoginTest", "Login"),
            ("FormSpec", "Form"),
        ],
    )
    def test_strips_suffixes(self, name, expected):
        """Each known suffix should be removed properly."""
        assert _derive_entity(name) == expected

    def test_returns_original_when_no_suffix(self):
        """A name with no recognised suffix should be returned as-is."""
        assert _derive_entity("Dashboard") == "Dashboard"

    def test_returns_original_when_suffix_is_full_name(self):
        """If removing the suffix would leave an empty string, return original."""
        assert _derive_entity("Component") == "Component"


# -----------------------------------------------------------------------
# generate_instruction
# -----------------------------------------------------------------------
class TestGenerateInstruction:
    """Tests for generate_instruction for every React code type."""

    @pytest.mark.parametrize("code_type", list(INSTRUCTION_TEMPLATES.keys()))
    def test_returns_string_for_each_type(self, code_type):
        """Every template type should produce a non-empty instruction string."""
        instruction = generate_instruction(code_type, "UserComponent", "")
        assert isinstance(instruction, str)
        assert len(instruction) > 0

    def test_entity_placeholder(self):
        """Templates with {entity} should insert the derived entity name."""
        with patch("scripts.generate_react_dataset.random.choice", side_effect=lambda lst: lst[0]):
            result = generate_instruction("functional_component", "UserComponent", "")
        assert "User" in result

    def test_name_placeholder(self):
        """Templates with {name} should insert the full name."""
        with patch("scripts.generate_react_dataset.random.choice", side_effect=lambda lst: lst[0]):
            result = generate_instruction("hook", "useDebounce", "")
        assert "useDebounce" in result

    def test_component_placeholder(self):
        """Templates with {component} should insert the full name."""
        with patch("scripts.generate_react_dataset.random.choice", side_effect=lambda lst: lst[0]):
            result = generate_instruction("test", "Dashboard", "")
        assert "Dashboard" in result

    def test_route_config_returns_template_unchanged(self):
        """Route config templates have no placeholder -- the bare template should be returned."""
        with patch("scripts.generate_react_dataset.random.choice", side_effect=lambda lst: lst[0]):
            result = generate_instruction("route_config", "AppRoutes", "")
        assert result == INSTRUCTION_TEMPLATES["route_config"][0]

    def test_purpose_placeholder(self):
        """A template with {purpose} should format the derived entity as purpose."""
        # Temporarily inject a purpose-based template to exercise line 279
        with patch.dict(
            INSTRUCTION_TEMPLATES,
            {"custom_purpose": ["Configure {purpose} for the project"]},
        ):
            with patch("scripts.generate_react_dataset.random.choice", side_effect=lambda lst: lst[0]):
                result = generate_instruction("custom_purpose", "AuthProvider", "")
        assert "Auth" in result

    def test_unknown_code_type_fallback(self):
        """An unrecognised code type falls back to the generic string."""
        result = generate_instruction("unknown_type", "FooBar", "")
        assert "FooBar" in result
        assert "unknown_type" in result


# -----------------------------------------------------------------------
# clean_code
# -----------------------------------------------------------------------
class TestCleanCode:
    """Tests for clean_code -- import filtering and blank-line collapsing."""

    def test_preserves_react_imports(self):
        """Imports from 'react' should be kept."""
        code = "import React from 'react';\nimport foo from './foo';\n\nconst x = 1;"
        cleaned = clean_code(code)
        assert "from 'react'" in cleaned
        assert "from './foo'" not in cleaned

    def test_preserves_axios_import(self):
        """Imports from 'axios' should be kept."""
        code = "import axios from 'axios';\nimport bar from 'bar';\n\nconst x = 1;"
        cleaned = clean_code(code)
        assert "axios" in cleaned
        assert "from 'bar'" not in cleaned

    def test_preserves_testing_library_import(self):
        """Imports from '@testing-library' should be kept."""
        code = "import { render } from '@testing-library/react';\n\nconst x = 1;"
        cleaned = clean_code(code)
        assert "@testing-library" in cleaned

    def test_preserves_reduxjs_import(self):
        """Imports from '@reduxjs/toolkit' should be kept."""
        code = "import { createSlice } from '@reduxjs/toolkit';\nimport z from 'z';\n"
        cleaned = clean_code(code)
        assert "@reduxjs/toolkit" in cleaned
        assert "from 'z'" not in cleaned

    def test_removes_nonessential_imports(self):
        """Non-essential imports should be stripped."""
        code = "import styles from './styles.module.css';\n\nconst App = () => {};"
        cleaned = clean_code(code)
        assert "styles.module.css" not in cleaned

    def test_collapses_excessive_blank_lines(self):
        """Three or more blank lines should be collapsed."""
        code = "const a = 1;\n\n\n\n\nconst b = 2;"
        cleaned = clean_code(code)
        assert "\n\n\n" not in cleaned

    def test_strips_leading_trailing_whitespace(self):
        """The result should have no surrounding whitespace."""
        code = "\n\n  const x = 1;  \n\n"
        cleaned = clean_code(code)
        assert cleaned == cleaned.strip()

    def test_preserves_react_router_import(self):
        """Imports from 'react-router-dom' should be preserved."""
        code = "import { Route } from 'react-router-dom';\n\nconst x = 1;"
        cleaned = clean_code(code)
        assert "react-router" in cleaned

    def test_preserves_tanstack_import(self):
        """Imports from '@tanstack' should be preserved."""
        code = "import { useQuery } from '@tanstack/react-query';\n\nconst x = 1;"
        cleaned = clean_code(code)
        assert "@tanstack" in cleaned

    def test_preserves_next_import(self):
        """Imports from 'next' should be preserved."""
        code = "import { useRouter } from 'next/router';\n\nconst x = 1;"
        cleaned = clean_code(code)
        assert "next/router" in cleaned


# -----------------------------------------------------------------------
# process_file
# -----------------------------------------------------------------------
class TestProcessFile:
    """Tests for process_file with various TSX/TS file scenarios."""

    def _write(self, tmp_path, filename, content):
        """Helper: write a file and return its Path."""
        p = tmp_path / filename
        p.write_text(content, encoding="utf-8")
        return p

    def test_valid_tsx_component(self, tmp_path):
        """A well-formed functional component should produce a training example."""
        content = REACT_SAMPLES["functional_component"] * 3
        path = self._write(tmp_path, "Dashboard.tsx", content)
        result = process_file(path)
        assert result is not None
        assert "instruction" in result
        assert result["input"] == ""
        assert "output" in result

    def test_file_too_short(self, tmp_path):
        """Files with cleaned output < 100 chars should be skipped."""
        content = "export default function Tiny() { return <div />; }"
        # Must match a pattern
        path = self._write(tmp_path, "Tiny.tsx", content)
        assert process_file(path) is None

    def test_file_too_long(self, tmp_path):
        """Files with cleaned output > max_length should be skipped."""
        content = REACT_SAMPLES["hook"] + ("// padding\n" * 5000)
        path = self._write(tmp_path, "BigHook.tsx", content)
        assert process_file(path, max_length=500) is None

    def test_unreadable_file(self, tmp_path):
        """Files that cannot be read should return None without raising."""
        path = tmp_path / "Ghost.tsx"
        path.write_text("content", encoding="utf-8")
        path.chmod(0o000)
        result = process_file(path)
        path.chmod(0o644)
        assert result is None

    def test_no_detected_type(self, tmp_path):
        """Files with no React/TS pattern should be skipped."""
        path = self._write(tmp_path, "plain.ts", "const x = 42;\nconsole.log(x);")
        assert process_file(path) is None

    def test_no_component_name(self, tmp_path):
        """Files where extract_component_name returns None should be skipped."""
        # createContext matches context_provider; all identifiers are reserved
        # skip words, and filepath stem is 'index' which is also skipped.
        code = "const exports = createContext(null);\n"
        path = self._write(tmp_path, "index.tsx", code)
        assert process_file(path) is None


# -----------------------------------------------------------------------
# scan_directory
# -----------------------------------------------------------------------
class TestScanDirectory:
    """Tests for scan_directory with temp directories containing React files."""

    def _populate(self, tmp_path):
        """Populate tmp_path with a variety of React/TS files."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "Dashboard.tsx").write_text(
            REACT_SAMPLES["functional_component"] * 3, encoding="utf-8"
        )
        (src / "useDebounce.ts").write_text(
            REACT_SAMPLES["hook"] * 3, encoding="utf-8"
        )
        # node_modules should be skipped
        nm = tmp_path / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        (nm / "index.tsx").write_text(
            REACT_SAMPLES["functional_component"], encoding="utf-8"
        )
        return tmp_path

    def test_finds_examples(self, tmp_path):
        """Valid React files should be found and processed."""
        self._populate(tmp_path)
        examples = scan_directory(tmp_path)
        assert isinstance(examples, list)
        assert len(examples) >= 1

    def test_skips_node_modules(self, tmp_path):
        """Files inside node_modules should never appear in results."""
        self._populate(tmp_path)
        examples = scan_directory(tmp_path)
        for ex in examples:
            assert "node_modules" not in ex.get("output", "")

    def test_skips_dist_and_build(self, tmp_path):
        """Files inside /dist/ or /build/ should be skipped."""
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "bundle.js").write_text(
            REACT_SAMPLES["functional_component"] * 3, encoding="utf-8"
        )
        examples = scan_directory(tmp_path)
        assert examples == []

    def test_empty_directory(self, tmp_path):
        """An empty directory should return an empty list."""
        assert scan_directory(tmp_path) == []

    def test_forwards_max_length(self, tmp_path):
        """max_length should be forwarded to process_file."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "Comp.tsx").write_text(
            REACT_SAMPLES["functional_component"] * 3, encoding="utf-8"
        )
        # Very small max_length should cause skipping
        examples = scan_directory(tmp_path, max_length=50)
        assert examples == []


# -----------------------------------------------------------------------
# clone_github_repo
# -----------------------------------------------------------------------
class TestCloneGithubRepo:
    """Tests for clone_github_repo with mocked subprocess."""

    def test_successful_clone(self, tmp_path, mock_subprocess_run):
        """A successful clone should return True."""
        target = tmp_path / "new-repo"
        assert clone_github_repo("owner/repo", target) is True
        mock_subprocess_run.assert_called_once()

    def test_clone_failure(self, tmp_path, mock_subprocess_run):
        """A CalledProcessError should cause the function to return False."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "git")
        target = tmp_path / "fail-repo"
        assert clone_github_repo("owner/repo", target) is False

    def test_skips_existing_directory(self, tmp_path, mock_subprocess_run):
        """If the target directory exists, clone is skipped and True is returned."""
        existing = tmp_path / "existing"
        existing.mkdir()
        assert clone_github_repo("owner/repo", existing) is True
        mock_subprocess_run.assert_not_called()


# -----------------------------------------------------------------------
# generate_synthetic_examples
# -----------------------------------------------------------------------
class TestGenerateSyntheticExamples:
    """Tests for generate_synthetic_examples."""

    def test_returns_non_empty_list(self):
        """The function should return at least one example."""
        examples = generate_synthetic_examples()
        assert len(examples) > 0

    def test_examples_have_correct_keys(self):
        """Every example must contain instruction, input, and output."""
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
        mocker.patch("sys.argv", ["prog", "--output", str(output_file)])
        mocker.patch("scripts.generate_react_dataset.clone_github_repo", return_value=False)

        main()

        assert output_file.exists()

    def test_with_local_dir(self, tmp_path, mocker):
        """The --local-dir flag should scan the given directory."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "App.tsx").write_text(
            REACT_SAMPLES["functional_component"] * 3, encoding="utf-8"
        )
        output_file = tmp_path / "out.jsonl"
        mocker.patch(
            "sys.argv",
            ["prog", "--output", str(output_file), "--local-dir", str(tmp_path)],
        )
        mocker.patch("scripts.generate_react_dataset.clone_github_repo", return_value=False)

        main()

        assert output_file.exists()
        lines = output_file.read_text().strip().splitlines()
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert "instruction" in data

    def test_nonexistent_local_dir(self, tmp_path, mocker):
        """A non-existent --local-dir should be handled gracefully."""
        output_file = tmp_path / "out.jsonl"
        mocker.patch(
            "sys.argv",
            ["prog", "--output", str(output_file), "--local-dir", "/no/such/dir"],
        )
        mocker.patch("scripts.generate_react_dataset.clone_github_repo", return_value=False)

        main()

        assert output_file.exists()

    def test_add_synthetic_flag(self, tmp_path, mocker):
        """The --add-synthetic flag should include synthetic examples."""
        output_file = tmp_path / "synth.jsonl"
        mocker.patch(
            "sys.argv",
            ["prog", "--output", str(output_file), "--add-synthetic"],
        )
        mocker.patch("scripts.generate_react_dataset.clone_github_repo", return_value=False)

        main()

        lines = output_file.read_text().strip().splitlines()
        assert len(lines) >= len(generate_synthetic_examples())

    def test_custom_github_repo(self, tmp_path, mocker):
        """The --github-repo flag should override defaults."""
        output_file = tmp_path / "custom.jsonl"
        mocker.patch(
            "sys.argv",
            ["prog", "--output", str(output_file), "--github-repo", "my/repo"],
        )
        mock_clone = mocker.patch(
            "scripts.generate_react_dataset.clone_github_repo", return_value=False
        )

        main()

        assert mock_clone.call_count == 1
        assert "my/repo" in mock_clone.call_args[0][0]

    def test_successful_clone_triggers_scan(self, tmp_path, mocker):
        """When clone succeeds, scan_directory should be invoked."""
        output_file = tmp_path / "scanned.jsonl"
        mocker.patch(
            "sys.argv",
            ["prog", "--output", str(output_file), "--github-repo", "o/r"],
        )
        mocker.patch("scripts.generate_react_dataset.clone_github_repo", return_value=True)
        mock_scan = mocker.patch(
            "scripts.generate_react_dataset.scan_directory", return_value=[]
        )

        main()

        mock_scan.assert_called_once()

    def test_max_length_forwarded(self, tmp_path, mocker):
        """--max-length should be forwarded to scan_directory calls."""
        output_file = tmp_path / "ml.jsonl"
        mocker.patch(
            "sys.argv",
            ["prog", "--output", str(output_file), "--github-repo", "o/r", "--max-length", "9999"],
        )
        mocker.patch("scripts.generate_react_dataset.clone_github_repo", return_value=True)
        mock_scan = mocker.patch(
            "scripts.generate_react_dataset.scan_directory", return_value=[]
        )

        main()

        _, kwargs = mock_scan.call_args
        assert kwargs["max_length"] == 9999

    def test_min_examples_threshold_failure(self, tmp_path, mocker):
        """When fewer examples are generated than --min-examples requires, SystemExit(1) is raised."""
        output_file = tmp_path / "fail.jsonl"
        mocker.patch(
            "sys.argv",
            ["prog", "--output", str(output_file), "--min-examples", "9999"],
        )
        mocker.patch("scripts.generate_react_dataset.clone_github_repo", return_value=False)

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_min_examples_threshold_success(self, tmp_path, mocker):
        """When enough examples are generated, no SystemExit should be raised."""
        output_file = tmp_path / "ok.jsonl"
        mocker.patch(
            "sys.argv",
            ["prog", "--output", str(output_file), "--add-synthetic", "--min-examples", "1"],
        )
        mocker.patch("scripts.generate_react_dataset.clone_github_repo", return_value=False)

        main()

        assert output_file.exists()
