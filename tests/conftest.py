"""Shared pytest fixtures for the bridge-cli test suite."""

import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Sample Java code strings -- one per CODE_PATTERN type
# ---------------------------------------------------------------------------

SAMPLE_CONTROLLER_JAVA = textwrap.dedent("""\
    package com.example.demo.controller;

    import org.springframework.web.bind.annotation.RestController;
    import org.springframework.web.bind.annotation.GetMapping;

    @RestController
    public class UserController {

        @GetMapping("/users")
        public String getUsers() {
            return "users";
        }
    }
""")

SAMPLE_SERVICE_JAVA = textwrap.dedent("""\
    package com.example.demo.service;

    import org.springframework.stereotype.Service;

    @Service
    public class OrderService {

        public void processOrder(Long orderId) {
            // business logic
        }
    }
""")

SAMPLE_REPOSITORY_JAVA = textwrap.dedent("""\
    package com.example.demo.repository;

    import org.springframework.stereotype.Repository;

    @Repository
    public interface ProductRepository extends JpaRepository<Product, Long> {
        List<Product> findByCategory(String category);
    }
""")

SAMPLE_ENTITY_JAVA = textwrap.dedent("""\
    package com.example.demo.entity;

    import javax.persistence.Entity;
    import javax.persistence.Id;

    @Entity
    public class Invoice {
        @Id
        private Long id;
        private String description;
    }
""")

SAMPLE_CONFIG_JAVA = textwrap.dedent("""\
    package com.example.demo.config;

    import org.springframework.context.annotation.Configuration;

    @Configuration
    public class SecurityConfig {

        // configuration beans
    }
""")

SAMPLE_TEST_JAVA = textwrap.dedent("""\
    package com.example.demo;

    import org.springframework.boot.test.context.SpringBootTest;

    @SpringBootTest
    public class ApplicationTest {

        void contextLoads() {}
    }
""")

SAMPLE_SECURITY_JAVA = textwrap.dedent("""\
    package com.example.demo.security;

    import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;

    @EnableWebSecurity
    public class WebSecurityConfig {
        // security filter chain
    }
""")

SAMPLE_SCHEDULING_JAVA = textwrap.dedent("""\
    package com.example.demo.tasks;

    import org.springframework.scheduling.annotation.Scheduled;

    public class ReportScheduler {
        @Scheduled(cron = "0 0 * * * *")
        public void generateReport() {}
    }
""")

SAMPLE_CACHING_JAVA = textwrap.dedent("""\
    package com.example.demo.cache;

    import org.springframework.cache.annotation.Cacheable;

    public class CachedProductService {
        @Cacheable("products")
        public String find() { return "product"; }
    }
""")

SAMPLE_WEBSOCKET_JAVA = textwrap.dedent("""\
    package com.example.demo.ws;

    import org.springframework.messaging.handler.annotation.MessageMapping;

    public class ChatController {
        @MessageMapping("/chat")
        public void handleMessage(String msg) {}
    }
""")

SAMPLE_VALIDATION_JAVA = textwrap.dedent("""\
    package com.example.demo.validation;

    import javax.validation.Valid;

    public class UserValidator {
        public void validate(@Valid Object o) {}
    }
""")

SAMPLE_ERROR_HANDLING_JAVA = textwrap.dedent("""\
    package com.example.demo.error;

    import org.springframework.web.bind.annotation.ExceptionHandler;
    import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

    public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {
        @ExceptionHandler(RuntimeException.class)
        public void handle(RuntimeException ex) {}
    }
""")

SAMPLE_AOP_JAVA = textwrap.dedent("""\
    package com.example.demo.aop;

    import org.aspectj.lang.annotation.Aspect;
    import org.aspectj.lang.annotation.Around;

    @Aspect
    public class LoggingAspect {
        @Around("execution(* com.example..*(..))")
        public Object log(org.aspectj.lang.ProceedingJoinPoint jp) throws Throwable {
            return jp.proceed();
        }
    }
""")

SAMPLE_MESSAGING_JAVA = textwrap.dedent("""\
    package com.example.demo.kafka;

    import org.springframework.kafka.annotation.KafkaListener;

    public class OrderListener {
        @KafkaListener(topics = "orders")
        public void listen(String msg) {}
    }
""")

SAMPLE_DOCKER = textwrap.dedent("""\
    FROM eclipse-temurin:17-jre-alpine
    WORKDIR /app
    COPY target/*.jar app.jar
    ENTRYPOINT ["java", "-jar", "app.jar"]
""")

SAMPLE_OPENAPI_JAVA = textwrap.dedent("""\
    package com.example.demo.config;

    import io.swagger.v3.oas.annotations.OpenAPIDefinition;

    @OpenAPIDefinition
    public class OpenApiConfig {}
""")

# Map of code type -> sample source
JAVA_SAMPLES = {
    "controller": SAMPLE_CONTROLLER_JAVA,
    "service": SAMPLE_SERVICE_JAVA,
    "repository": SAMPLE_REPOSITORY_JAVA,
    "entity": SAMPLE_ENTITY_JAVA,
    "config": SAMPLE_CONFIG_JAVA,
    "test": SAMPLE_TEST_JAVA,
    "security": SAMPLE_SECURITY_JAVA,
    "scheduling": SAMPLE_SCHEDULING_JAVA,
    "caching": SAMPLE_CACHING_JAVA,
    "websocket": SAMPLE_WEBSOCKET_JAVA,
    "validation": SAMPLE_VALIDATION_JAVA,
    "error_handling": SAMPLE_ERROR_HANDLING_JAVA,
    "aop": SAMPLE_AOP_JAVA,
    "messaging": SAMPLE_MESSAGING_JAVA,
    "docker": SAMPLE_DOCKER,
    "openapi": SAMPLE_OPENAPI_JAVA,
}


@pytest.fixture()
def sample_java_sources():
    """Return a dict mapping each CODE_PATTERN type to a sample Java source string."""
    return dict(JAVA_SAMPLES)


@pytest.fixture()
def java_temp_dir(tmp_path):
    """Create a temporary directory tree with several sample Java files.

    The tree looks like::

        tmp_path/
            src/main/java/
                UserController.java   (controller, repeated for length)
                OrderService.java     (service, repeated for length)
                Invoice.java          (entity, repeated for length)
            src/test/java/
                ApplicationTest.java  (test -- lives under /test/, should be skipped)
    """
    main_dir = tmp_path / "src" / "main" / "java"
    main_dir.mkdir(parents=True)

    test_dir = tmp_path / "src" / "test" / "java"
    test_dir.mkdir(parents=True)

    (main_dir / "UserController.java").write_text(
        SAMPLE_CONTROLLER_JAVA * 5, encoding="utf-8"
    )
    (main_dir / "OrderService.java").write_text(
        SAMPLE_SERVICE_JAVA * 5, encoding="utf-8"
    )
    (main_dir / "Invoice.java").write_text(
        SAMPLE_ENTITY_JAVA * 5, encoding="utf-8"
    )
    # File under /test/ path -- should be skipped by scan_directory
    (test_dir / "ApplicationTest.java").write_text(
        SAMPLE_TEST_JAVA * 5, encoding="utf-8"
    )

    return tmp_path


@pytest.fixture()
def mock_subprocess_run(mocker):
    """Mock subprocess.run to simulate git clone behaviour.

    Returns the mock so tests can inspect calls or change side_effect.
    """
    return mocker.patch("subprocess.run")
