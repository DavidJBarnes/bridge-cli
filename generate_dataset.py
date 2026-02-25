#!/usr/bin/env python3
"""
Spring Boot Dataset Generator for Bridge CLI
Generates instruction-response pairs from Spring Boot code repositories

Usage:
    python generate_dataset.py --output datasets/spring-boot-extended.jsonl
    python generate_dataset.py --github-repo in28minutes/spring-boot-examples
"""

import json
import os
import re
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import random

# Template patterns for generating instructions from code
INSTRUCTION_TEMPLATES = {
    "controller": [
        "Create a Spring Boot REST controller for {entity}",
        "Write a Spring Boot controller with CRUD endpoints for {entity}",
        "Implement a RESTful API controller for managing {entity} in Spring Boot",
    ],
    "service": [
        "Create a Spring Boot service class for {entity} business logic",
        "Write a Spring Boot service with transaction management for {entity}",
        "Implement a service layer for {entity} operations in Spring Boot",
    ],
    "repository": [
        "Write a Spring Data JPA repository for {entity}",
        "Create a JPA repository with custom query methods for {entity}",
        "Implement a Spring Boot repository interface for {entity}",
    ],
    "entity": [
        "Create a JPA entity class for {entity}",
        "Write a Spring Boot entity with validation and relationships for {entity}",
        "Define a JPA entity with auditing for {entity}",
    ],
    "config": [
        "Create a Spring Boot configuration class for {purpose}",
        "Write a Spring configuration for {purpose}",
        "Implement Spring Boot configuration for {purpose}",
    ],
    "test": [
        "Write a Spring Boot integration test for {component}",
        "Create a unit test for Spring Boot {component}",
        "Implement MockMvc tests for {component}",
    ],
}

# Patterns to identify code type
CODE_PATTERNS = {
    "controller": r"@(Rest)?Controller",
    "service": r"@Service",
    "repository": r"@Repository|extends (Jpa|Crud)Repository",
    "entity": r"@Entity",
    "config": r"@Configuration",
    "test": r"@SpringBootTest|@WebMvcTest|@DataJpaTest",
}


def detect_code_type(content: str) -> Optional[str]:
    """Detect the type of Spring Boot component from code content."""
    for code_type, pattern in CODE_PATTERNS.items():
        if re.search(pattern, content):
            return code_type
    return None


def extract_class_name(content: str) -> Optional[str]:
    """Extract the main class name from Java code."""
    match = re.search(r"public\s+(?:class|interface)\s+(\w+)", content)
    return match.group(1) if match else None


def extract_entity_name(class_name: str) -> str:
    """Extract entity name from class name (e.g., UserController -> User)."""
    suffixes = ["Controller", "Service", "Repository", "Impl", "Test", "Config"]
    name = class_name
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name if name else class_name


def generate_instruction(code_type: str, class_name: str, content: str) -> str:
    """Generate a natural instruction for the given code."""
    entity = extract_entity_name(class_name)

    if code_type in INSTRUCTION_TEMPLATES:
        template = random.choice(INSTRUCTION_TEMPLATES[code_type])
        if "{entity}" in template:
            return template.format(entity=entity)
        elif "{purpose}" in template:
            # Try to infer purpose from class name
            purpose = class_name.replace("Config", " configuration").strip()
            return template.format(purpose=purpose)
        elif "{component}" in template:
            return template.format(component=class_name)
    return f"Create a Spring Boot {code_type} class similar to {class_name}"


def clean_code(content: str) -> str:
    """Clean and normalize Java code."""
    # Remove package declarations
    content = re.sub(r"^package\s+[\w.]+;\s*\n", "", content)
    # Remove most imports (keep essential ones in output)
    content = re.sub(r"^import\s+[\w.*]+;\s*\n", "", content, flags=re.MULTILINE)
    # Remove excessive blank lines
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def process_java_file(filepath: Path) -> Optional[Dict]:
    """Process a single Java file and create a training example."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    code_type = detect_code_type(content)
    if not code_type:
        return None

    class_name = extract_class_name(content)
    if not class_name:
        return None

    # Skip test files for now (can be enabled)
    if code_type == "test":
        return None

    instruction = generate_instruction(code_type, class_name, content)
    output = clean_code(content)

    # Skip very short or very long files
    if len(output) < 200 or len(output) > 15000:
        return None

    return {"instruction": instruction, "input": "", "output": output}


def scan_directory(directory: Path) -> List[Dict]:
    """Scan directory for Java files and generate training examples."""
    examples = []
    java_files = list(directory.rglob("*.java"))

    print(f"Found {len(java_files)} Java files")

    for filepath in java_files:
        # Skip test directories if desired
        if "/test/" in str(filepath):
            continue

        example = process_java_file(filepath)
        if example:
            examples.append(example)
            print(f"  ✓ Processed: {filepath.name}")

    return examples


def clone_github_repo(repo_url: str, target_dir: Path) -> bool:
    """Clone a GitHub repository."""
    if target_dir.exists():
        print(f"Directory {target_dir} already exists, skipping clone")
        return True

    try:
        cmd = ["git", "clone", "--depth", "1", f"https://github.com/{repo_url}.git", str(target_dir)]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Cloned {repo_url}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone {repo_url}: {e}")
        return False


def generate_synthetic_examples() -> List[Dict]:
    """Generate additional synthetic Spring Boot examples."""
    synthetic = []

    # Add more examples programmatically
    examples = [
        {
            "instruction": "Create a Spring Boot DTO class with validation for user registration",
            "input": "",
            "output": """@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UserRegistrationDTO {

    @NotBlank(message = "Email is required")
    @Email(message = "Please provide a valid email address")
    private String email;

    @NotBlank(message = "Password is required")
    @Size(min = 8, max = 100, message = "Password must be between 8 and 100 characters")
    @Pattern(regexp = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\\\\d).*$",
             message = "Password must contain at least one uppercase, one lowercase, and one digit")
    private String password;

    @NotBlank(message = "First name is required")
    @Size(max = 50)
    private String firstName;

    @NotBlank(message = "Last name is required")
    @Size(max = 50)
    private String lastName;

    @Pattern(regexp = "^\\\\+?[1-9]\\\\d{1,14}$", message = "Invalid phone number")
    private String phoneNumber;
}""",
        },
        {
            "instruction": "Write a Spring Boot pagination response wrapper class",
            "input": "",
            "output": """@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PageResponse<T> {

    private List<T> content;
    private int pageNumber;
    private int pageSize;
    private long totalElements;
    private int totalPages;
    private boolean first;
    private boolean last;

    public static <T> PageResponse<T> from(Page<T> page) {
        return PageResponse.<T>builder()
                .content(page.getContent())
                .pageNumber(page.getNumber())
                .pageSize(page.getSize())
                .totalElements(page.getTotalElements())
                .totalPages(page.getTotalPages())
                .first(page.isFirst())
                .last(page.isLast())
                .build();
    }

    public static <T, R> PageResponse<R> from(Page<T> page, Function<T, R> mapper) {
        List<R> mappedContent = page.getContent().stream()
                .map(mapper)
                .collect(Collectors.toList());
        
        return PageResponse.<R>builder()
                .content(mappedContent)
                .pageNumber(page.getNumber())
                .pageSize(page.getSize())
                .totalElements(page.getTotalElements())
                .totalPages(page.getTotalPages())
                .first(page.isFirst())
                .last(page.isLast())
                .build();
    }
}""",
        },
        {
            "instruction": "Create Spring Boot application properties for production environment",
            "input": "",
            "output": """# Application
spring.application.name=my-spring-boot-app
server.port=${PORT:8080}

# Database
spring.datasource.url=jdbc:postgresql://${DB_HOST}:${DB_PORT}/${DB_NAME}
spring.datasource.username=${DB_USERNAME}
spring.datasource.password=${DB_PASSWORD}
spring.datasource.driver-class-name=org.postgresql.Driver

# HikariCP
spring.datasource.hikari.maximum-pool-size=20
spring.datasource.hikari.minimum-idle=5
spring.datasource.hikari.idle-timeout=300000
spring.datasource.hikari.max-lifetime=1200000
spring.datasource.hikari.connection-timeout=20000

# JPA/Hibernate
spring.jpa.hibernate.ddl-auto=validate
spring.jpa.show-sql=false
spring.jpa.properties.hibernate.dialect=org.hibernate.dialect.PostgreSQLDialect
spring.jpa.properties.hibernate.format_sql=false
spring.jpa.open-in-view=false

# Logging
logging.level.root=WARN
logging.level.com.example=INFO
logging.pattern.console=%d{yyyy-MM-dd HH:mm:ss} [%thread] %-5level %logger{36} - %msg%n

# Actuator
management.endpoints.web.exposure.include=health,info,metrics,prometheus
management.endpoint.health.show-details=when_authorized

# Security
spring.security.user.name=${ADMIN_USERNAME:admin}
spring.security.user.password=${ADMIN_PASSWORD}

# Jackson
spring.jackson.serialization.write-dates-as-timestamps=false
spring.jackson.default-property-inclusion=non_null""",
        },
    ]

    synthetic.extend(examples)
    return synthetic


def main():
    parser = argparse.ArgumentParser(description="Generate Spring Boot training dataset")
    parser.add_argument("--output", "-o", default="datasets/spring-boot-extended.jsonl", help="Output file path")
    parser.add_argument("--github-repo", "-g", action="append", help="GitHub repo to process (owner/repo format)")
    parser.add_argument("--local-dir", "-d", help="Local directory to scan")
    parser.add_argument("--add-synthetic", action="store_true", help="Add synthetic examples")
    args = parser.parse_args()

    all_examples = []

    # Default repos to process
    default_repos = [
        "in28minutes/spring-boot-examples",
        "RameshMF/spring-boot-tutorial",
        "spring-projects/spring-data-examples",
    ]

    repos = args.github_repo if args.github_repo else default_repos

    # Process GitHub repos
    temp_dir = Path("/tmp/spring-boot-repos")
    temp_dir.mkdir(exist_ok=True)

    for repo in repos:
        repo_name = repo.split("/")[-1]
        repo_dir = temp_dir / repo_name

        if clone_github_repo(repo, repo_dir):
            examples = scan_directory(repo_dir)
            all_examples.extend(examples)
            print(f"Extracted {len(examples)} examples from {repo}")

    # Process local directory if specified
    if args.local_dir:
        local_path = Path(args.local_dir)
        if local_path.exists():
            examples = scan_directory(local_path)
            all_examples.extend(examples)
            print(f"Extracted {len(examples)} examples from local directory")

    # Add synthetic examples
    if args.add_synthetic:
        synthetic = generate_synthetic_examples()
        all_examples.extend(synthetic)
        print(f"Added {len(synthetic)} synthetic examples")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    print(f"\n✅ Generated {len(all_examples)} training examples")
    print(f"📁 Output saved to: {output_path}")


if __name__ == "__main__":
    main()
