#!/usr/bin/env python3
"""React/TypeScript Dataset Generator for Bridge CLI.

Generates instruction-response pairs from React and TypeScript code repositories
for fine-tuning language models. Mirrors the Java/Spring Boot generator at
scripts/generate_dataset.py but targets the React/TypeScript ecosystem.

Supported component types:
    - Functional components (forms, tables, lists, modals, dashboards)
    - Custom hooks (useAuth, useFetch, useForm, useDebounce, etc.)
    - Context providers (auth, theme, notification)
    - Higher-order components (HOCs)
    - React Router configurations
    - API service layers (axios/fetch wrappers, interceptors)
    - Redux slices and stores
    - Test files (React Testing Library, Jest)
    - TypeScript interfaces and type definitions

Usage:
    python generate_react_dataset.py --output datasets/react-extended.jsonl
    python generate_react_dataset.py --github-repo alan2207/bulletproof-react
    python generate_react_dataset.py --add-synthetic --min-examples 50
"""

import argparse
import json
import logging
import os
import random
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

__all__ = [
    "CODE_PATTERNS",
    "INSTRUCTION_TEMPLATES",
    "detect_code_type",
    "extract_component_name",
    "generate_instruction",
    "clean_code",
    "process_file",
    "scan_directory",
    "clone_github_repo",
    "generate_synthetic_examples",
    "main",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns used to classify React/TypeScript source files
# ---------------------------------------------------------------------------

CODE_PATTERNS: Dict[str, str] = {
    "functional_component": (
        r"(?:export\s+(?:default\s+)?)?(?:const|function)\s+\w+\s*"
        r"(?::\s*React\.FC|:\s*FC)?"
        r"[^=]*=?\s*(?:\([^)]*\))?\s*(?::\s*JSX\.Element)?\s*(?:=>)?\s*\{"
        r"[\s\S]*?(?:return\s*\(?\s*<|<\w)"
    ),
    "hook": r"(?:export\s+)?(?:const|function)\s+use[A-Z]\w*\s*(?:=\s*)?(?:\([^)]*\))?\s*(?:=>)?\s*\{",
    "context_provider": (
        r"(?:createContext|React\.createContext)\s*(?:<[^>]*>)?\s*\("
    ),
    "hoc": r"(?:export\s+)?(?:const|function)\s+with[A-Z]\w*\s*(?:=\s*)?\(",
    "route_config": (
        r"(?:<(?:BrowserRouter|Routes|Route|Switch)|"
        r"createBrowserRouter|createRoutesFromElements)"
    ),
    "api_service": (
        r"(?:axios\.(?:create|get|post|put|delete|patch)|"
        r"fetch\s*\(\s*[`'\"]|"
        r"(?:export\s+)?(?:const|class)\s+\w*(?:Api|Service|Client)\b)"
    ),
    "redux_slice": (
        r"(?:createSlice|createAsyncThunk|configureStore|createStore|combineReducers)"
    ),
    "test": (
        r"(?:describe|it|test)\s*\(\s*['\"]|"
        r"(?:render|screen|fireEvent|waitFor|userEvent)\b.*(?:@testing-library|react)"
    ),
    "typescript_types": (
        r"(?:export\s+)?(?:interface|type)\s+\w+\s*(?:<[^>]*>)?\s*(?:=\s*)?[{\(]"
    ),
}

# ---------------------------------------------------------------------------
# Instruction templates -- at least 3 per detected type
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATES: Dict[str, List[str]] = {
    "functional_component": [
        "Create a React functional component for a {entity} with TypeScript",
        "Write a reusable React component that renders a {entity}",
        "Implement a React {entity} component with proper prop typing",
        "Build a responsive {entity} component using React and TypeScript",
        "Design a {entity} React component with accessibility support",
    ],
    "hook": [
        "Create a custom React hook called {name}",
        "Write a reusable custom hook {name} with TypeScript generics",
        "Implement the {name} custom hook with proper cleanup and error handling",
        "Build a {name} hook that can be shared across React components",
    ],
    "context_provider": [
        "Create a React context provider for {entity}",
        "Write a typed React context with provider and consumer hook for {entity}",
        "Implement a {entity} context provider with reducer-based state management",
        "Build a {entity} context that supports nested providers in React",
    ],
    "hoc": [
        "Create a higher-order component (HOC) called {name} in React",
        "Write a React HOC {name} with proper TypeScript type forwarding",
        "Implement a {name} HOC that injects props into wrapped components",
    ],
    "route_config": [
        "Create a React Router configuration with protected routes",
        "Write a route configuration with lazy loading and code splitting",
        "Implement a React Router setup with nested layouts and guards",
        "Build a routing configuration with role-based access control in React",
    ],
    "api_service": [
        "Create a typed API service layer for {entity} using axios",
        "Write an API client with interceptors and error handling for {entity}",
        "Implement a fetch wrapper service for {entity} REST endpoints",
        "Build a typed HTTP service for {entity} with request/response transformations",
    ],
    "redux_slice": [
        "Create a Redux Toolkit slice for {entity} state management",
        "Write a Redux slice with async thunks for {entity}",
        "Implement a {entity} Redux store with selectors and middleware",
        "Build a Redux Toolkit slice for {entity} with optimistic updates",
    ],
    "test": [
        "Write React Testing Library tests for the {component} component",
        "Create integration tests for {component} using Jest and RTL",
        "Implement unit tests for {component} with user event simulation",
        "Write comprehensive test coverage for {component} with mocking",
    ],
    "typescript_types": [
        "Define TypeScript interfaces for a {entity} REST API",
        "Create TypeScript type definitions for {entity} data models",
        "Write strict TypeScript types and interfaces for the {entity} domain",
    ],
}

# ---------------------------------------------------------------------------
# File extensions to consider
# ---------------------------------------------------------------------------

_REACT_EXTENSIONS = {".tsx", ".ts", ".jsx", ".js"}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def detect_code_type(content: str) -> Optional[str]:
    """Detect the type of React/TypeScript component from file content.

    Args:
        content: The full text content of a source file.

    Returns:
        A string key from ``CODE_PATTERNS`` if a pattern matches, or ``None``
        if the content does not match any known React/TypeScript pattern.
    """
    # Check in a specific order: more specific patterns first to avoid
    # false positives (e.g. a test file also contains JSX).
    priority_order = [
        "test",
        "redux_slice",
        "route_config",
        "hook",
        "context_provider",
        "hoc",
        "api_service",
        "typescript_types",
        "functional_component",
    ]
    for code_type in priority_order:
        pattern = CODE_PATTERNS[code_type]
        if re.search(pattern, content):
            return code_type
    return None


def extract_component_name(content: str, filepath: Optional[Path] = None) -> Optional[str]:
    """Extract the primary component or export name from React/TypeScript code.

    The function tries multiple strategies in order:
        1. ``export default function ComponentName``
        2. ``export default ComponentName``
        3. ``export const ComponentName``
        4. ``function ComponentName``
        5. ``const ComponentName``
        6. Falls back to the file stem (without extension) if *filepath* is given.

    Args:
        content: The full text content of a source file.
        filepath: Optional path used as a fallback for name extraction.

    Returns:
        The extracted component/export name, or ``None`` if nothing could be
        determined.
    """
    patterns = [
        r"export\s+default\s+function\s+(\w+)",
        r"export\s+default\s+(\w+)",
        r"export\s+(?:const|function)\s+(\w+)",
        r"function\s+(\w+)\s*\(",
        r"const\s+(\w+)\s*(?::\s*\w+)?\s*=",
    ]
    for pat in patterns:
        match = re.search(pat, content)
        if match:
            name = match.group(1)
            # Skip common non-component names
            if name not in {"default", "module", "exports", "require", "undefined"}:
                return name

    if filepath is not None:
        stem = filepath.stem
        # Remove common suffixes like .test, .spec, .stories
        stem = re.sub(r"\.(test|spec|stories|styles|types)$", "", stem)
        if stem and stem != "index":
            return stem

    return None


def _derive_entity(name: str) -> str:
    """Derive a human-friendly entity name from a component/file name.

    Args:
        name: A PascalCase or camelCase identifier.

    Returns:
        The name with common suffixes stripped, preserving the domain word.
    """
    suffixes = [
        "Component", "Container", "Provider", "Context", "Slice",
        "Service", "Client", "Api", "Hook", "HOC", "Page", "View",
        "Screen", "Test", "Spec",
    ]
    result = name
    for suffix in suffixes:
        if result.endswith(suffix) and result != suffix:
            result = result[: -len(suffix)]
            break
    return result if result else name


def generate_instruction(code_type: str, name: str, content: str) -> str:
    """Generate a natural-language instruction for the given code.

    Args:
        code_type: One of the keys from ``CODE_PATTERNS``.
        name: The component/export name extracted from the source.
        content: The full source text (used for additional heuristics).

    Returns:
        A human-readable instruction string suitable for the ``instruction``
        field in an Alpaca-format training example.
    """
    entity = _derive_entity(name)

    if code_type in INSTRUCTION_TEMPLATES:
        template = random.choice(INSTRUCTION_TEMPLATES[code_type])
        if "{entity}" in template:
            return template.format(entity=entity)
        if "{name}" in template:
            return template.format(name=name)
        if "{component}" in template:
            return template.format(component=name)
        if "{purpose}" in template:
            return template.format(purpose=entity)
        return template

    return f"Create a React/TypeScript {code_type} similar to {name}"


def clean_code(content: str) -> str:
    """Clean and normalise React/TypeScript source code for training output.

    Removes excessive blank lines and trims leading/trailing whitespace while
    preserving essential React imports (``react``, ``react-dom``,
    ``react-router-dom``, ``@reduxjs/toolkit``, ``axios``, ``@testing-library``).

    Args:
        content: Raw source file text.

    Returns:
        Cleaned source text ready for inclusion in a training example.
    """
    lines = content.splitlines()
    kept_lines: List[str] = []

    # Imports we want to preserve in the output
    _essential_import_patterns = [
        r"from\s+['\"]react['\"]",
        r"from\s+['\"]react-dom",
        r"from\s+['\"]react-router",
        r"from\s+['\"]@reduxjs/toolkit",
        r"from\s+['\"]axios",
        r"from\s+['\"]@testing-library",
        r"from\s+['\"]@tanstack",
        r"from\s+['\"]next",
    ]
    essential_re = re.compile("|".join(_essential_import_patterns))

    for line in lines:
        # Remove non-essential import lines
        if re.match(r"^\s*import\s+", line):
            if essential_re.search(line):
                kept_lines.append(line)
            # else: skip this import
        else:
            kept_lines.append(line)

    result = "\n".join(kept_lines)
    # Collapse runs of 3+ blank lines into 2
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def process_file(filepath: Path, max_length: int = 15000) -> Optional[Dict[str, str]]:
    """Process a single React/TypeScript file and create a training example.

    Args:
        filepath: Absolute or relative path to a ``.tsx``, ``.ts``, ``.jsx``,
            or ``.js`` source file.
        max_length: Maximum character length for the cleaned output. Files
            exceeding this limit are skipped.

    Returns:
        An Alpaca-format dict ``{"instruction": ..., "input": "", "output": ...}``
        or ``None`` if the file should be skipped.
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read %s: %s", filepath, exc)
        return None

    code_type = detect_code_type(content)
    if code_type is None:
        logger.debug("No recognised pattern in %s", filepath.name)
        return None

    name = extract_component_name(content, filepath)
    if name is None:
        logger.debug("Could not extract name from %s", filepath.name)
        return None

    instruction = generate_instruction(code_type, name, content)
    output = clean_code(content)

    if len(output) < 100:
        logger.debug("Skipping %s -- output too short (%d chars)", filepath.name, len(output))
        return None

    if len(output) > max_length:
        logger.debug("Skipping %s -- output too long (%d chars)", filepath.name, len(output))
        return None

    return {"instruction": instruction, "input": "", "output": output}


def scan_directory(directory: Path, max_length: int = 15000) -> List[Dict[str, str]]:
    """Scan a directory tree for React/TypeScript files and generate examples.

    Args:
        directory: Root directory to scan recursively.
        max_length: Passed through to :func:`process_file`.

    Returns:
        A list of Alpaca-format training example dicts.
    """
    examples: List[Dict[str, str]] = []
    all_files: List[Path] = []

    for ext in _REACT_EXTENSIONS:
        all_files.extend(directory.rglob(f"*{ext}"))

    logger.info("Found %d React/TypeScript files in %s", len(all_files), directory)

    for filepath in sorted(all_files):
        # Skip common non-source directories
        parts_str = str(filepath)
        if any(skip in parts_str for skip in ["/node_modules/", "/dist/", "/build/", "/.next/", "/.cache/"]):
            continue

        example = process_file(filepath, max_length=max_length)
        if example is not None:
            examples.append(example)
            logger.info("  Processed: %s", filepath.name)

    return examples


def clone_github_repo(repo_url: str, target_dir: Path) -> bool:
    """Clone a GitHub repository with shallow depth.

    Args:
        repo_url: Repository identifier in ``owner/repo`` format.
        target_dir: Local filesystem path to clone into.

    Returns:
        ``True`` if the clone succeeded or the directory already exists,
        ``False`` on failure.
    """
    if target_dir.exists():
        logger.info("Directory %s already exists, skipping clone", target_dir)
        return True

    try:
        cmd = [
            "git", "clone", "--depth", "1",
            f"https://github.com/{repo_url}.git",
            str(target_dir),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info("Cloned %s", repo_url)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to clone %s: %s", repo_url, exc)
        return False


# ---------------------------------------------------------------------------
# Synthetic examples
# ---------------------------------------------------------------------------


def generate_synthetic_examples() -> List[Dict[str, str]]:
    """Generate high-quality synthetic React/TypeScript training examples.

    Returns:
        A list of at least 10 Alpaca-format training examples covering
        API services, hooks, components, routing, Redux, context providers,
        tests, TypeScript types, and error boundaries.
    """
    return [
        # 1 -- Typed API service with axios
        {
            "instruction": "Create a typed API service layer using axios with interceptors and error handling in TypeScript",
            "input": "",
            "output": """\
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, InternalAxiosRequestConfig } from 'axios';

interface ApiError {
  status: number;
  message: string;
  errors?: Record<string, string[]>;
}

interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

class ApiService {
  private client: AxiosInstance;

  constructor(baseURL: string = '/api') {
    this.client = axios.create({
      baseURL,
      timeout: 15000,
      headers: { 'Content-Type': 'application/json' },
    });

    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        const token = localStorage.getItem('auth_token');
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error),
    );

    this.client.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
        }
        const apiError: ApiError = {
          status: error.response?.status ?? 500,
          message: error.response?.data?.message ?? 'An unexpected error occurred',
          errors: error.response?.data?.errors,
        };
        return Promise.reject(apiError);
      },
    );
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  async post<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  async put<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  async patch<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.patch<T>(url, data, config);
    return response.data;
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }

  async getPaginated<T>(
    url: string,
    page: number = 1,
    pageSize: number = 20,
    params?: Record<string, unknown>,
  ): Promise<PaginatedResponse<T>> {
    return this.get<PaginatedResponse<T>>(url, {
      params: { page, pageSize, ...params },
    });
  }
}

export const apiService = new ApiService();
export default ApiService;""",
        },
        # 2 -- useAuth custom hook
        {
            "instruction": "Create a useAuth custom hook for authentication with login, logout, and token refresh in React TypeScript",
            "input": "",
            "output": """\
import { useState, useCallback, useEffect, useRef } from 'react';

interface User {
  id: string;
  email: string;
  name: string;
  roles: string[];
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

interface LoginCredentials {
  email: string;
  password: string;
}

interface AuthResponse {
  user: User;
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

interface UseAuthReturn extends AuthState {
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
  clearError: () => void;
}

const TOKEN_KEY = 'auth_token';
const REFRESH_KEY = 'refresh_token';
const USER_KEY = 'auth_user';

export function useAuth(): UseAuthReturn {
  const [state, setState] = useState<AuthState>(() => {
    const token = localStorage.getItem(TOKEN_KEY);
    const storedUser = localStorage.getItem(USER_KEY);
    return {
      user: storedUser ? JSON.parse(storedUser) : null,
      token,
      isAuthenticated: !!token,
      isLoading: false,
      error: null,
    };
  });

  const refreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const scheduleRefresh = useCallback((expiresIn: number) => {
    if (refreshTimerRef.current) {
      clearTimeout(refreshTimerRef.current);
    }
    const refreshTime = (expiresIn - 60) * 1000;
    refreshTimerRef.current = setTimeout(() => {
      refreshTokenFn();
    }, Math.max(refreshTime, 0));
  }, []);

  const login = useCallback(async (credentials: LoginCredentials) => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Login failed');
      }

      const data: AuthResponse = await response.json();
      localStorage.setItem(TOKEN_KEY, data.accessToken);
      localStorage.setItem(REFRESH_KEY, data.refreshToken);
      localStorage.setItem(USER_KEY, JSON.stringify(data.user));

      setState({
        user: data.user,
        token: data.accessToken,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      });

      scheduleRefresh(data.expiresIn);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Login failed';
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: message,
      }));
    }
  }, [scheduleRefresh]);

  const refreshTokenFn = useCallback(async () => {
    const storedRefresh = localStorage.getItem(REFRESH_KEY);
    if (!storedRefresh) return;

    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refreshToken: storedRefresh }),
      });

      if (!response.ok) throw new Error('Token refresh failed');

      const data: AuthResponse = await response.json();
      localStorage.setItem(TOKEN_KEY, data.accessToken);
      localStorage.setItem(REFRESH_KEY, data.refreshToken);

      setState((prev) => ({
        ...prev,
        token: data.accessToken,
        user: data.user,
      }));

      scheduleRefresh(data.expiresIn);
    } catch {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(REFRESH_KEY);
      localStorage.removeItem(USER_KEY);
      setState({
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
      });
    }
  }, [scheduleRefresh]);

  const logout = useCallback(() => {
    if (refreshTimerRef.current) {
      clearTimeout(refreshTimerRef.current);
    }
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(REFRESH_KEY);
    localStorage.removeItem(USER_KEY);
    setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
    });
  }, []);

  const clearError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  useEffect(() => {
    return () => {
      if (refreshTimerRef.current) {
        clearTimeout(refreshTimerRef.current);
      }
    };
  }, []);

  return {
    ...state,
    login,
    logout,
    refreshToken: refreshTokenFn,
    clearError,
  };
}

export default useAuth;""",
        },
        # 3 -- Data table component with pagination
        {
            "instruction": "Build a reusable data table component with sorting, pagination, and column configuration in React TypeScript",
            "input": "",
            "output": """\
import React, { useState, useMemo, useCallback } from 'react';

interface Column<T> {
  key: keyof T & string;
  header: string;
  sortable?: boolean;
  render?: (value: T[keyof T], row: T) => React.ReactNode;
  width?: string;
}

interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  pageSize?: number;
  onRowClick?: (row: T) => void;
  isLoading?: boolean;
  emptyMessage?: string;
}

type SortDirection = 'asc' | 'desc';

interface SortConfig {
  key: string;
  direction: SortDirection;
}

function DataTable<T extends Record<string, unknown>>({
  data,
  columns,
  pageSize = 10,
  onRowClick,
  isLoading = false,
  emptyMessage = 'No data available',
}: DataTableProps<T>): JSX.Element {
  const [currentPage, setCurrentPage] = useState(1);
  const [sortConfig, setSortConfig] = useState<SortConfig | null>(null);

  const sortedData = useMemo(() => {
    if (!sortConfig) return data;

    return [...data].sort((a, b) => {
      const aVal = a[sortConfig.key];
      const bVal = b[sortConfig.key];

      if (aVal === bVal) return 0;
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;

      const comparison = aVal < bVal ? -1 : 1;
      return sortConfig.direction === 'asc' ? comparison : -comparison;
    });
  }, [data, sortConfig]);

  const totalPages = Math.ceil(sortedData.length / pageSize);
  const startIndex = (currentPage - 1) * pageSize;
  const paginatedData = sortedData.slice(startIndex, startIndex + pageSize);

  const handleSort = useCallback((key: string) => {
    setSortConfig((prev) => {
      if (prev?.key === key) {
        return prev.direction === 'asc'
          ? { key, direction: 'desc' }
          : null;
      }
      return { key, direction: 'asc' };
    });
    setCurrentPage(1);
  }, []);

  const handlePageChange = useCallback((page: number) => {
    setCurrentPage(Math.max(1, Math.min(page, totalPages)));
  }, [totalPages]);

  if (isLoading) {
    return (
      <div className="data-table-loading" role="status" aria-label="Loading data">
        <div className="spinner" />
        <span>Loading...</span>
      </div>
    );
  }

  return (
    <div className="data-table-container">
      <table className="data-table" role="grid">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                style={{ width: col.width }}
                onClick={col.sortable ? () => handleSort(col.key) : undefined}
                className={col.sortable ? 'sortable' : ''}
                aria-sort={
                  sortConfig?.key === col.key
                    ? sortConfig.direction === 'asc'
                      ? 'ascending'
                      : 'descending'
                    : undefined
                }
              >
                {col.header}
                {col.sortable && sortConfig?.key === col.key && (
                  <span className="sort-indicator">
                    {sortConfig.direction === 'asc' ? ' \\u25B2' : ' \\u25BC'}
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {paginatedData.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="empty-message">
                {emptyMessage}
              </td>
            </tr>
          ) : (
            paginatedData.map((row, idx) => (
              <tr
                key={idx}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
                className={onRowClick ? 'clickable' : ''}
                tabIndex={onRowClick ? 0 : undefined}
                onKeyDown={(e) => {
                  if (onRowClick && (e.key === 'Enter' || e.key === ' ')) {
                    e.preventDefault();
                    onRowClick(row);
                  }
                }}
              >
                {columns.map((col) => (
                  <td key={col.key}>
                    {col.render
                      ? col.render(row[col.key], row)
                      : String(row[col.key] ?? '')}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>

      {totalPages > 1 && (
        <nav className="pagination" aria-label="Table pagination">
          <button
            onClick={() => handlePageChange(1)}
            disabled={currentPage === 1}
            aria-label="First page"
          >
            &laquo;
          </button>
          <button
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage === 1}
            aria-label="Previous page"
          >
            &lsaquo;
          </button>
          <span className="page-info">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
            aria-label="Next page"
          >
            &rsaquo;
          </button>
          <button
            onClick={() => handlePageChange(totalPages)}
            disabled={currentPage === totalPages}
            aria-label="Last page"
          >
            &raquo;
          </button>
        </nav>
      )}
    </div>
  );
}

export default DataTable;""",
        },
        # 4 -- Form component with validation
        {
            "instruction": "Create a React form component with field validation, error display, and submission handling in TypeScript",
            "input": "",
            "output": """\
import React, { useState, useCallback, FormEvent } from 'react';

interface ValidationRule {
  required?: boolean;
  minLength?: number;
  maxLength?: number;
  pattern?: RegExp;
  custom?: (value: string) => string | null;
}

interface FieldConfig {
  name: string;
  label: string;
  type?: string;
  placeholder?: string;
  rules?: ValidationRule;
}

interface FormValues {
  [key: string]: string;
}

interface FormErrors {
  [key: string]: string;
}

interface ValidatedFormProps {
  fields: FieldConfig[];
  onSubmit: (values: FormValues) => Promise<void>;
  submitLabel?: string;
  className?: string;
}

function validateField(value: string, rules?: ValidationRule): string | null {
  if (!rules) return null;

  if (rules.required && !value.trim()) {
    return 'This field is required';
  }
  if (rules.minLength && value.length < rules.minLength) {
    return `Must be at least ${rules.minLength} characters`;
  }
  if (rules.maxLength && value.length > rules.maxLength) {
    return `Must be no more than ${rules.maxLength} characters`;
  }
  if (rules.pattern && !rules.pattern.test(value)) {
    return 'Invalid format';
  }
  if (rules.custom) {
    return rules.custom(value);
  }
  return null;
}

export default function ValidatedForm({
  fields,
  onSubmit,
  submitLabel = 'Submit',
  className = '',
}: ValidatedFormProps): JSX.Element {
  const [values, setValues] = useState<FormValues>(() =>
    Object.fromEntries(fields.map((f) => [f.name, ''])),
  );
  const [errors, setErrors] = useState<FormErrors>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleChange = useCallback(
    (name: string, value: string) => {
      setValues((prev) => ({ ...prev, [name]: value }));
      setSubmitError(null);

      if (touched[name]) {
        const field = fields.find((f) => f.name === name);
        const error = validateField(value, field?.rules);
        setErrors((prev) => {
          const next = { ...prev };
          if (error) {
            next[name] = error;
          } else {
            delete next[name];
          }
          return next;
        });
      }
    },
    [fields, touched],
  );

  const handleBlur = useCallback(
    (name: string) => {
      setTouched((prev) => ({ ...prev, [name]: true }));
      const field = fields.find((f) => f.name === name);
      const error = validateField(values[name], field?.rules);
      setErrors((prev) => {
        const next = { ...prev };
        if (error) {
          next[name] = error;
        } else {
          delete next[name];
        }
        return next;
      });
    },
    [fields, values],
  );

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      setSubmitError(null);

      // Validate all fields
      const newErrors: FormErrors = {};
      const allTouched: Record<string, boolean> = {};
      for (const field of fields) {
        allTouched[field.name] = true;
        const error = validateField(values[field.name], field.rules);
        if (error) {
          newErrors[field.name] = error;
        }
      }
      setTouched(allTouched);
      setErrors(newErrors);

      if (Object.keys(newErrors).length > 0) return;

      setIsSubmitting(true);
      try {
        await onSubmit(values);
      } catch (err) {
        setSubmitError(
          err instanceof Error ? err.message : 'Submission failed',
        );
      } finally {
        setIsSubmitting(false);
      }
    },
    [fields, values, onSubmit],
  );

  return (
    <form onSubmit={handleSubmit} className={`validated-form ${className}`} noValidate>
      {fields.map((field) => (
        <div key={field.name} className="form-field">
          <label htmlFor={field.name}>{field.label}</label>
          <input
            id={field.name}
            name={field.name}
            type={field.type ?? 'text'}
            placeholder={field.placeholder}
            value={values[field.name]}
            onChange={(e) => handleChange(field.name, e.target.value)}
            onBlur={() => handleBlur(field.name)}
            aria-invalid={!!(touched[field.name] && errors[field.name])}
            aria-describedby={errors[field.name] ? `${field.name}-error` : undefined}
            disabled={isSubmitting}
          />
          {touched[field.name] && errors[field.name] && (
            <span id={`${field.name}-error`} className="field-error" role="alert">
              {errors[field.name]}
            </span>
          )}
        </div>
      ))}

      {submitError && (
        <div className="submit-error" role="alert">
          {submitError}
        </div>
      )}

      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? 'Submitting...' : submitLabel}
      </button>
    </form>
  );
}""",
        },
        # 5 -- React Router config with protected routes
        {
            "instruction": "Implement a React Router configuration with protected routes, role-based guards, and lazy-loaded pages",
            "input": "",
            "output": """\
import React, { Suspense, lazy } from 'react';
import { createBrowserRouter, Navigate, Outlet, RouterProvider } from 'react-router-dom';

// Lazy-loaded pages
const LoginPage = lazy(() => import('./pages/LoginPage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const UsersPage = lazy(() => import('./pages/UsersPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const ProfilePage = lazy(() => import('./pages/ProfilePage'));
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'));

interface User {
  id: string;
  roles: string[];
}

function useAuth(): { user: User | null; isLoading: boolean } {
  // Placeholder -- real implementation would come from AuthContext
  return { user: null, isLoading: false };
}

function LoadingFallback(): JSX.Element {
  return (
    <div className="loading-container" role="status">
      <div className="spinner" />
      <span>Loading page...</span>
    </div>
  );
}

interface ProtectedRouteProps {
  requiredRoles?: string[];
}

function ProtectedRoute({ requiredRoles }: ProtectedRouteProps): JSX.Element {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return <LoadingFallback />;
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (requiredRoles && requiredRoles.length > 0) {
    const hasRequiredRole = requiredRoles.some((role) =>
      user.roles.includes(role),
    );
    if (!hasRequiredRole) {
      return <Navigate to="/unauthorized" replace />;
    }
  }

  return (
    <Suspense fallback={<LoadingFallback />}>
      <Outlet />
    </Suspense>
  );
}

function PublicOnlyRoute(): JSX.Element {
  const { user } = useAuth();

  if (user) {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <Suspense fallback={<LoadingFallback />}>
      <Outlet />
    </Suspense>
  );
}

function RootLayout(): JSX.Element {
  return (
    <div className="app-layout">
      <header className="app-header">
        <nav>{/* Navigation links */}</nav>
      </header>
      <main className="app-main">
        <Suspense fallback={<LoadingFallback />}>
          <Outlet />
        </Suspense>
      </main>
      <footer className="app-footer">
        <p>Application Footer</p>
      </footer>
    </div>
  );
}

const router = createBrowserRouter([
  {
    path: '/',
    element: <RootLayout />,
    children: [
      {
        index: true,
        element: <Navigate to="/dashboard" replace />,
      },
      // Public-only routes (redirect to dashboard if authenticated)
      {
        element: <PublicOnlyRoute />,
        children: [
          { path: 'login', element: <LoginPage /> },
        ],
      },
      // Protected routes (require authentication)
      {
        element: <ProtectedRoute />,
        children: [
          { path: 'dashboard', element: <DashboardPage /> },
          { path: 'profile', element: <ProfilePage /> },
        ],
      },
      // Admin-only routes
      {
        element: <ProtectedRoute requiredRoles={['admin']} />,
        children: [
          { path: 'users', element: <UsersPage /> },
          { path: 'settings', element: <SettingsPage /> },
        ],
      },
      // Catch-all
      { path: '*', element: <NotFoundPage /> },
    ],
  },
]);

export default function AppRouter(): JSX.Element {
  return <RouterProvider router={router} />;
}""",
        },
        # 6 -- Redux slice for auth
        {
            "instruction": "Create a Redux Toolkit slice for authentication with async thunks for login, logout, and token refresh",
            "input": "",
            "output": """\
import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

interface User {
  id: string;
  email: string;
  name: string;
  roles: string[];
  avatarUrl?: string;
}

interface AuthState {
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

interface LoginPayload {
  email: string;
  password: string;
}

interface AuthTokens {
  user: User;
  accessToken: string;
  refreshToken: string;
}

const initialState: AuthState = {
  user: null,
  accessToken: localStorage.getItem('access_token'),
  refreshToken: localStorage.getItem('refresh_token'),
  isAuthenticated: !!localStorage.getItem('access_token'),
  isLoading: false,
  error: null,
};

export const loginUser = createAsyncThunk<AuthTokens, LoginPayload, { rejectValue: string }>(
  'auth/login',
  async (credentials, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials),
      });

      if (!response.ok) {
        const data = await response.json();
        return rejectWithValue(data.message || 'Login failed');
      }

      const data: AuthTokens = await response.json();
      localStorage.setItem('access_token', data.accessToken);
      localStorage.setItem('refresh_token', data.refreshToken);
      return data;
    } catch {
      return rejectWithValue('Network error. Please try again.');
    }
  },
);

export const refreshAccessToken = createAsyncThunk<AuthTokens, void, { rejectValue: string }>(
  'auth/refresh',
  async (_, { getState, rejectWithValue }) => {
    const state = getState() as { auth: AuthState };
    const { refreshToken } = state.auth;

    if (!refreshToken) {
      return rejectWithValue('No refresh token available');
    }

    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refreshToken }),
      });

      if (!response.ok) {
        return rejectWithValue('Token refresh failed');
      }

      const data: AuthTokens = await response.json();
      localStorage.setItem('access_token', data.accessToken);
      localStorage.setItem('refresh_token', data.refreshToken);
      return data;
    } catch {
      return rejectWithValue('Network error during token refresh');
    }
  },
);

export const logoutUser = createAsyncThunk<void, void>(
  'auth/logout',
  async () => {
    try {
      await fetch('/api/auth/logout', { method: 'POST' });
    } finally {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
    }
  },
);

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    clearError(state) {
      state.error = null;
    },
    updateUser(state, action: PayloadAction<Partial<User>>) {
      if (state.user) {
        state.user = { ...state.user, ...action.payload };
      }
    },
  },
  extraReducers: (builder) => {
    builder
      // Login
      .addCase(loginUser.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(loginUser.fulfilled, (state, action) => {
        state.isLoading = false;
        state.isAuthenticated = true;
        state.user = action.payload.user;
        state.accessToken = action.payload.accessToken;
        state.refreshToken = action.payload.refreshToken;
        state.error = null;
      })
      .addCase(loginUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload ?? 'Login failed';
      })
      // Refresh
      .addCase(refreshAccessToken.fulfilled, (state, action) => {
        state.accessToken = action.payload.accessToken;
        state.refreshToken = action.payload.refreshToken;
        state.user = action.payload.user;
      })
      .addCase(refreshAccessToken.rejected, (state) => {
        state.user = null;
        state.accessToken = null;
        state.refreshToken = null;
        state.isAuthenticated = false;
      })
      // Logout
      .addCase(logoutUser.fulfilled, (state) => {
        state.user = null;
        state.accessToken = null;
        state.refreshToken = null;
        state.isAuthenticated = false;
        state.error = null;
      });
  },
});

export const { clearError, updateUser } = authSlice.actions;

// Selectors
export const selectCurrentUser = (state: { auth: AuthState }) => state.auth.user;
export const selectIsAuthenticated = (state: { auth: AuthState }) => state.auth.isAuthenticated;
export const selectAuthLoading = (state: { auth: AuthState }) => state.auth.isLoading;
export const selectAuthError = (state: { auth: AuthState }) => state.auth.error;

export default authSlice.reducer;""",
        },
        # 7 -- Notification context provider
        {
            "instruction": "Create a notification context provider with auto-dismiss, stacking, and different severity levels in React TypeScript",
            "input": "",
            "output": """\
import React, { createContext, useContext, useReducer, useCallback, useRef, useEffect } from 'react';

type NotificationSeverity = 'success' | 'error' | 'warning' | 'info';

interface Notification {
  id: string;
  message: string;
  severity: NotificationSeverity;
  duration: number;
  createdAt: number;
}

interface NotificationState {
  notifications: Notification[];
}

type NotificationAction =
  | { type: 'ADD'; payload: Notification }
  | { type: 'REMOVE'; payload: string };

interface NotificationContextValue {
  notifications: Notification[];
  addNotification: (message: string, severity?: NotificationSeverity, duration?: number) => void;
  removeNotification: (id: string) => void;
  success: (message: string, duration?: number) => void;
  error: (message: string, duration?: number) => void;
  warning: (message: string, duration?: number) => void;
  info: (message: string, duration?: number) => void;
}

const NotificationContext = createContext<NotificationContextValue | undefined>(undefined);

const MAX_NOTIFICATIONS = 5;

function notificationReducer(
  state: NotificationState,
  action: NotificationAction,
): NotificationState {
  switch (action.type) {
    case 'ADD': {
      const updated = [action.payload, ...state.notifications];
      return { notifications: updated.slice(0, MAX_NOTIFICATIONS) };
    }
    case 'REMOVE':
      return {
        notifications: state.notifications.filter((n) => n.id !== action.payload),
      };
    default:
      return state;
  }
}

let idCounter = 0;
function generateId(): string {
  idCounter += 1;
  return `notification-${Date.now()}-${idCounter}`;
}

interface NotificationProviderProps {
  children: React.ReactNode;
}

export function NotificationProvider({ children }: NotificationProviderProps): JSX.Element {
  const [state, dispatch] = useReducer(notificationReducer, { notifications: [] });
  const timersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const removeNotification = useCallback((id: string) => {
    dispatch({ type: 'REMOVE', payload: id });
    const timer = timersRef.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timersRef.current.delete(id);
    }
  }, []);

  const addNotification = useCallback(
    (message: string, severity: NotificationSeverity = 'info', duration: number = 5000) => {
      const id = generateId();
      const notification: Notification = {
        id,
        message,
        severity,
        duration,
        createdAt: Date.now(),
      };

      dispatch({ type: 'ADD', payload: notification });

      if (duration > 0) {
        const timer = setTimeout(() => {
          removeNotification(id);
        }, duration);
        timersRef.current.set(id, timer);
      }
    },
    [removeNotification],
  );

  const success = useCallback(
    (message: string, duration?: number) => addNotification(message, 'success', duration),
    [addNotification],
  );
  const error = useCallback(
    (message: string, duration?: number) => addNotification(message, 'error', duration ?? 8000),
    [addNotification],
  );
  const warning = useCallback(
    (message: string, duration?: number) => addNotification(message, 'warning', duration),
    [addNotification],
  );
  const info = useCallback(
    (message: string, duration?: number) => addNotification(message, 'info', duration),
    [addNotification],
  );

  useEffect(() => {
    const timers = timersRef.current;
    return () => {
      timers.forEach((timer) => clearTimeout(timer));
      timers.clear();
    };
  }, []);

  const value: NotificationContextValue = {
    notifications: state.notifications,
    addNotification,
    removeNotification,
    success,
    error,
    warning,
    info,
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
      <div className="notification-container" aria-live="polite" aria-label="Notifications">
        {state.notifications.map((notification) => (
          <div
            key={notification.id}
            className={`notification notification-${notification.severity}`}
            role="alert"
          >
            <span className="notification-message">{notification.message}</span>
            <button
              className="notification-dismiss"
              onClick={() => removeNotification(notification.id)}
              aria-label="Dismiss notification"
            >
              &times;
            </button>
          </div>
        ))}
      </div>
    </NotificationContext.Provider>
  );
}

export function useNotifications(): NotificationContextValue {
  const context = useContext(NotificationContext);
  if (context === undefined) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
}

export default NotificationProvider;""",
        },
        # 8 -- React Testing Library test
        {
            "instruction": "Write comprehensive React Testing Library tests for a login form component with validation and submission",
            "input": "",
            "output": """\
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import LoginForm from './LoginForm';

// Mock the auth service
const mockLogin = jest.fn();

jest.mock('../hooks/useAuth', () => ({
  useAuth: () => ({
    login: mockLogin,
    isLoading: false,
    error: null,
  }),
}));

describe('LoginForm', () => {
  beforeEach(() => {
    mockLogin.mockReset();
  });

  it('renders email and password fields with labels', () => {
    render(<LoginForm />);

    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  it('shows validation errors when submitting empty form', async () => {
    const user = userEvent.setup();
    render(<LoginForm />);

    await user.click(screen.getByRole('button', { name: /sign in/i }));

    expect(await screen.findByText(/email is required/i)).toBeInTheDocument();
    expect(await screen.findByText(/password is required/i)).toBeInTheDocument();
    expect(mockLogin).not.toHaveBeenCalled();
  });

  it('shows error for invalid email format', async () => {
    const user = userEvent.setup();
    render(<LoginForm />);

    await user.type(screen.getByLabelText(/email/i), 'not-an-email');
    await user.tab();

    expect(await screen.findByText(/valid email/i)).toBeInTheDocument();
  });

  it('shows error when password is too short', async () => {
    const user = userEvent.setup();
    render(<LoginForm />);

    await user.type(screen.getByLabelText(/password/i), 'abc');
    await user.tab();

    expect(await screen.findByText(/at least 8 characters/i)).toBeInTheDocument();
  });

  it('submits the form with valid credentials', async () => {
    mockLogin.mockResolvedValueOnce({ success: true });
    const user = userEvent.setup();
    render(<LoginForm />);

    await user.type(screen.getByLabelText(/email/i), 'user@example.com');
    await user.type(screen.getByLabelText(/password/i), 'SecurePass123');
    await user.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith({
        email: 'user@example.com',
        password: 'SecurePass123',
      });
    });
  });

  it('displays server error message on login failure', async () => {
    mockLogin.mockRejectedValueOnce(new Error('Invalid credentials'));
    const user = userEvent.setup();
    render(<LoginForm />);

    await user.type(screen.getByLabelText(/email/i), 'user@example.com');
    await user.type(screen.getByLabelText(/password/i), 'WrongPassword1');
    await user.click(screen.getByRole('button', { name: /sign in/i }));

    expect(await screen.findByRole('alert')).toHaveTextContent(/invalid credentials/i);
  });

  it('disables the submit button while loading', async () => {
    mockLogin.mockImplementation(
      () => new Promise((resolve) => setTimeout(resolve, 1000)),
    );
    const user = userEvent.setup();
    render(<LoginForm />);

    await user.type(screen.getByLabelText(/email/i), 'user@example.com');
    await user.type(screen.getByLabelText(/password/i), 'SecurePass123');
    await user.click(screen.getByRole('button', { name: /sign in/i }));

    expect(screen.getByRole('button', { name: /signing in/i })).toBeDisabled();
  });

  it('clears field error when user starts typing', async () => {
    const user = userEvent.setup();
    render(<LoginForm />);

    // Trigger validation
    await user.click(screen.getByRole('button', { name: /sign in/i }));
    expect(await screen.findByText(/email is required/i)).toBeInTheDocument();

    // Start typing to clear the error
    await user.type(screen.getByLabelText(/email/i), 'u');

    await waitFor(() => {
      expect(screen.queryByText(/email is required/i)).not.toBeInTheDocument();
    });
  });

  it('is accessible with proper ARIA attributes', () => {
    render(<LoginForm />);

    const emailInput = screen.getByLabelText(/email/i);
    const passwordInput = screen.getByLabelText(/password/i);

    expect(emailInput).toHaveAttribute('type', 'email');
    expect(passwordInput).toHaveAttribute('type', 'password');
    expect(emailInput).toHaveAttribute('aria-required', 'true');
    expect(passwordInput).toHaveAttribute('aria-required', 'true');
  });
});""",
        },
        # 9 -- TypeScript interfaces for a REST API
        {
            "instruction": "Define comprehensive TypeScript interfaces and types for a REST API with users, posts, and comments",
            "input": "",
            "output": """\
// --- Base types ---

export interface Timestamps {
  createdAt: string;
  updatedAt: string;
}

export interface PaginationParams {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
    hasNextPage: boolean;
    hasPreviousPage: boolean;
  };
}

export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
}

export interface ApiErrorResponse {
  success: false;
  message: string;
  errorCode: string;
  errors?: FieldError[];
}

export interface FieldError {
  field: string;
  message: string;
  code: string;
}

// --- User domain ---

export type UserRole = 'admin' | 'editor' | 'viewer';
export type UserStatus = 'active' | 'inactive' | 'suspended';

export interface User extends Timestamps {
  id: string;
  email: string;
  name: string;
  avatarUrl: string | null;
  role: UserRole;
  status: UserStatus;
  lastLoginAt: string | null;
}

export interface CreateUserRequest {
  email: string;
  name: string;
  password: string;
  role?: UserRole;
}

export interface UpdateUserRequest {
  name?: string;
  email?: string;
  avatarUrl?: string | null;
  role?: UserRole;
  status?: UserStatus;
}

export type UserSummary = Pick<User, 'id' | 'name' | 'avatarUrl'>;

// --- Post domain ---

export type PostStatus = 'draft' | 'published' | 'archived';

export interface Post extends Timestamps {
  id: string;
  title: string;
  slug: string;
  content: string;
  excerpt: string;
  coverImageUrl: string | null;
  status: PostStatus;
  publishedAt: string | null;
  author: UserSummary;
  tags: string[];
  commentCount: number;
}

export interface CreatePostRequest {
  title: string;
  content: string;
  excerpt?: string;
  coverImageUrl?: string;
  status?: PostStatus;
  tags?: string[];
}

export interface UpdatePostRequest {
  title?: string;
  content?: string;
  excerpt?: string;
  coverImageUrl?: string | null;
  status?: PostStatus;
  tags?: string[];
}

export interface PostFilters extends PaginationParams {
  status?: PostStatus;
  authorId?: string;
  tag?: string;
  search?: string;
  publishedAfter?: string;
  publishedBefore?: string;
}

export type PostListItem = Omit<Post, 'content'>;

// --- Comment domain ---

export interface Comment extends Timestamps {
  id: string;
  postId: string;
  parentId: string | null;
  content: string;
  author: UserSummary;
  isEdited: boolean;
  replies?: Comment[];
}

export interface CreateCommentRequest {
  postId: string;
  parentId?: string;
  content: string;
}

export interface UpdateCommentRequest {
  content: string;
}

// --- Auth domain ---

export interface LoginRequest {
  email: string;
  password: string;
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
  tokenType: 'Bearer';
}

export interface LoginResponse extends AuthTokens {
  user: User;
}

export interface RefreshTokenRequest {
  refreshToken: string;
}

export interface ChangePasswordRequest {
  currentPassword: string;
  newPassword: string;
}""",
        },
        # 10 -- Error boundary component
        {
            "instruction": "Create a React error boundary component with fallback UI, error reporting, and retry functionality in TypeScript",
            "input": "",
            "output": """\
import React, { Component, ErrorInfo, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode | ((error: Error, resetError: () => void) => ReactNode);
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  onReset?: () => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo });

    // Report to external error tracking service
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('ErrorBoundary caught an error:', error);
      console.error('Component stack:', errorInfo.componentStack);
    }
  }

  resetError = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });

    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  render(): ReactNode {
    if (this.state.hasError && this.state.error) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        if (typeof this.props.fallback === 'function') {
          return this.props.fallback(this.state.error, this.resetError);
        }
        return this.props.fallback;
      }

      // Default fallback UI
      return (
        <div className="error-boundary" role="alert">
          <div className="error-boundary-content">
            <h2>Something went wrong</h2>
            <p className="error-message">
              {this.state.error.message || 'An unexpected error occurred'}
            </p>

            {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
              <details className="error-details">
                <summary>Error Details</summary>
                <pre className="error-stack">{this.state.error.stack}</pre>
                <pre className="component-stack">
                  {this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}

            <div className="error-actions">
              <button
                onClick={this.resetError}
                className="error-retry-button"
              >
                Try Again
              </button>
              <button
                onClick={() => window.location.reload()}
                className="error-reload-button"
              >
                Reload Page
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * HOC to wrap a component with an error boundary.
 */
export function withErrorBoundary<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  errorBoundaryProps?: Omit<ErrorBoundaryProps, 'children'>,
): React.FC<P> {
  const WithErrorBoundary: React.FC<P> = (props) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <WrappedComponent {...props} />
    </ErrorBoundary>
  );

  WithErrorBoundary.displayName = `withErrorBoundary(${
    WrappedComponent.displayName || WrappedComponent.name || 'Component'
  })`;

  return WithErrorBoundary;
}

export default ErrorBoundary;""",
        },
    ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and orchestrate dataset generation.

    Supports processing GitHub repositories, local directories, and synthetic
    example injection. Outputs Alpaca-format JSONL.
    """
    parser = argparse.ArgumentParser(
        description="Generate React/TypeScript training dataset for fine-tuning",
    )
    parser.add_argument(
        "--output", "-o",
        default="datasets/react-extended.jsonl",
        help="Output JSONL file path (default: datasets/react-extended.jsonl)",
    )
    parser.add_argument(
        "--github-repo", "-g",
        action="append",
        help="GitHub repo in owner/repo format (may be repeated)",
    )
    parser.add_argument(
        "--local-dir", "-d",
        help="Local directory to scan for React/TypeScript files",
    )
    parser.add_argument(
        "--add-synthetic",
        action="store_true",
        help="Include synthetic high-quality examples in the output",
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=0,
        help="Minimum number of examples required; exits with error if not met",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=15000,
        help="Maximum character length for code output (default: 15000)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    all_examples: List[Dict[str, str]] = []

    # Default repos -- popular React/TypeScript repositories
    default_repos = [
        "alan2207/bulletproof-react",
        "tiangolo/full-stack-fastapi-template",
        "refinedev/refine",
    ]

    repos = args.github_repo if args.github_repo else default_repos

    # Clone and scan GitHub repos
    temp_dir = Path("/tmp/react-repos")
    temp_dir.mkdir(exist_ok=True)

    for repo in repos:
        repo_name = repo.split("/")[-1]
        repo_dir = temp_dir / repo_name

        if clone_github_repo(repo, repo_dir):
            examples = scan_directory(repo_dir, max_length=args.max_length)
            all_examples.extend(examples)
            logger.info("Extracted %d examples from %s", len(examples), repo)

    # Process local directory if specified
    if args.local_dir:
        local_path = Path(args.local_dir)
        if local_path.exists():
            examples = scan_directory(local_path, max_length=args.max_length)
            all_examples.extend(examples)
            logger.info("Extracted %d examples from local directory", len(examples))
        else:
            logger.warning("Local directory does not exist: %s", args.local_dir)

    # Add synthetic examples
    if args.add_synthetic:
        synthetic = generate_synthetic_examples()
        all_examples.extend(synthetic)
        logger.info("Added %d synthetic examples", len(synthetic))

    # Check minimum example threshold
    if args.min_examples and len(all_examples) < args.min_examples:
        logger.error(
            "Only %d examples generated, but --min-examples requires %d",
            len(all_examples),
            args.min_examples,
        )
        raise SystemExit(1)

    # Write JSONL output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        for example in all_examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info("Generated %d training examples", len(all_examples))
    logger.info("Output saved to: %s", output_path)


if __name__ == "__main__":
    main()
