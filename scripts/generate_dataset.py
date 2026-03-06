#!/usr/bin/env python3
"""Spring Boot Dataset Generator for Bridge CLI.

Generates instruction-response pairs from Spring Boot code repositories
for fine-tuning language models. Supports scanning GitHub repos, local
directories, and includes synthetic examples covering Spring Boot patterns
such as controllers, services, repositories, security, caching, scheduling,
WebSocket, validation, error handling, AOP, messaging/Kafka, Docker/deployment,
and OpenAPI/Swagger configurations.

Usage::

    python generate_dataset.py --output datasets/spring-boot-extended.jsonl
    python generate_dataset.py --github-repo in28minutes/spring-boot-examples
    python generate_dataset.py --add-synthetic --min-examples 300 --max-length 20000
"""

import json
import logging
import os
import re
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import random

__all__ = [
    "INSTRUCTION_TEMPLATES",
    "CODE_PATTERNS",
    "detect_code_type",
    "extract_class_name",
    "extract_entity_name",
    "generate_instruction",
    "clean_code",
    "process_java_file",
    "scan_directory",
    "clone_github_repo",
    "generate_synthetic_examples",
    "main",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template patterns for generating instructions from code
# ---------------------------------------------------------------------------
INSTRUCTION_TEMPLATES: Dict[str, List[str]] = {
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
    "security": [
        "Create a Spring Security configuration for {purpose}",
        "Write a Spring Boot security filter chain for {purpose}",
        "Implement JWT-based authentication configuration for {purpose}",
    ],
    "scheduling": [
        "Create a Spring Boot scheduled task for {purpose}",
        "Write a cron-based scheduled job in Spring Boot for {purpose}",
        "Implement a fixed-rate scheduled service for {purpose}",
    ],
    "caching": [
        "Create a Spring Boot caching configuration for {purpose}",
        "Write a cache-enabled service for {entity} in Spring Boot",
        "Implement Redis cache configuration for {purpose}",
    ],
    "websocket": [
        "Create a Spring Boot WebSocket endpoint for {purpose}",
        "Write a STOMP WebSocket handler in Spring Boot for {purpose}",
        "Implement a WebSocket message broker configuration for {purpose}",
    ],
    "validation": [
        "Create a custom Bean Validation constraint for {entity}",
        "Write a Spring Boot request validation handler for {entity}",
        "Implement a custom validator for {entity} in Spring Boot",
    ],
    "error_handling": [
        "Create a global exception handler for a Spring Boot REST API",
        "Write a @ControllerAdvice error handler for {entity} operations",
        "Implement centralized error handling for {purpose}",
    ],
    "aop": [
        "Create a Spring AOP aspect for {purpose}",
        "Write a logging aspect using Spring AOP for {purpose}",
        "Implement a performance monitoring aspect for {purpose}",
    ],
    "messaging": [
        "Create a Spring Boot Kafka producer for {entity}",
        "Write a Kafka consumer service in Spring Boot for {entity}",
        "Implement a Spring Boot message listener for {purpose}",
    ],
    "docker": [
        "Create a Dockerfile for a Spring Boot application",
        "Write a docker-compose.yml for Spring Boot with {purpose}",
        "Implement a multi-stage Docker build for a Spring Boot project",
    ],
    "openapi": [
        "Create an OpenAPI/Swagger configuration for a Spring Boot REST API",
        "Write Swagger annotations for the {entity} controller",
        "Implement API documentation with SpringDoc OpenAPI for {purpose}",
    ],
}

# ---------------------------------------------------------------------------
# Patterns to identify code type
# ---------------------------------------------------------------------------
CODE_PATTERNS: Dict[str, str] = {
    "controller": r"@(Rest)?Controller",
    "service": r"@Service",
    "repository": r"@Repository|extends (Jpa|Crud)Repository",
    "entity": r"@Entity",
    "config": r"@Configuration",
    "test": r"@SpringBootTest|@WebMvcTest|@DataJpaTest",
    "security": r"@EnableWebSecurity|SecurityFilterChain|WebSecurityConfigurer",
    "scheduling": r"@Scheduled|@EnableScheduling",
    "caching": r"@EnableCaching|@Cacheable|@CacheEvict|@CachePut",
    "websocket": r"@EnableWebSocket|@MessageMapping|WebSocketConfigurer|StompEndpointRegistry",
    "validation": r"@Valid|@Validated|ConstraintValidator|@Pattern|@NotBlank",
    "error_handling": r"@ControllerAdvice|@ExceptionHandler|ResponseEntityExceptionHandler",
    "aop": r"@Aspect|@Around|@Before|@After|@Pointcut",
    "messaging": r"@KafkaListener|KafkaTemplate|@EnableKafka|@JmsListener",
    "docker": r"^FROM\s+.*|docker-compose|Dockerfile",
    "openapi": r"@OpenAPIDefinition|@Operation|@ApiResponse|SpringDoc|springdoc",
}


def detect_code_type(content: str) -> Optional[str]:
    """Detect the type of Spring Boot component from code content.

    Iterates through ``CODE_PATTERNS`` and returns the first matching
    component type.

    Args:
        content: The raw source code text to analyse.

    Returns:
        A string key from ``CODE_PATTERNS`` if a match is found, or
        ``None`` when no recognisable pattern is detected.
    """
    for code_type, pattern in CODE_PATTERNS.items():
        if re.search(pattern, content):
            return code_type
    return None


def extract_class_name(content: str) -> Optional[str]:
    """Extract the main class or interface name from Java source code.

    Args:
        content: The raw Java source text.

    Returns:
        The identifier of the first ``public class`` or ``public interface``
        declaration found, or ``None`` if none is present.
    """
    match = re.search(r"public\s+(?:class|interface)\s+(\w+)", content)
    return match.group(1) if match else None


def extract_entity_name(class_name: str) -> str:
    """Derive a short entity name by stripping common Spring suffixes.

    For example, ``UserController`` becomes ``User``.

    Args:
        class_name: The full class name (e.g. ``OrderServiceImpl``).

    Returns:
        The entity name with the recognised suffix removed, or the
        original *class_name* if no suffix matched.
    """
    suffixes = [
        "Controller",
        "Service",
        "Repository",
        "Impl",
        "Test",
        "Config",
        "Aspect",
        "Handler",
        "Listener",
        "Producer",
        "Consumer",
        "Advisor",
    ]
    name = class_name
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name if name else class_name


def generate_instruction(code_type: str, class_name: str, content: str) -> str:
    """Generate a natural-language instruction for the given source code.

    Selects a random template from ``INSTRUCTION_TEMPLATES`` that matches
    *code_type* and fills in placeholder variables derived from
    *class_name*.

    Args:
        code_type: A key that exists in ``INSTRUCTION_TEMPLATES``.
        class_name: The Java class name extracted from the source file.
        content: The full source text (reserved for future heuristics).

    Returns:
        A human-readable instruction string suitable for use as a
        training prompt.
    """
    entity = extract_entity_name(class_name)

    if code_type in INSTRUCTION_TEMPLATES:
        template = random.choice(INSTRUCTION_TEMPLATES[code_type])
        if "{entity}" in template:
            return template.format(entity=entity)
        elif "{purpose}" in template:
            purpose = class_name.replace("Config", " configuration").strip()
            return template.format(purpose=purpose)
        elif "{component}" in template:
            return template.format(component=class_name)
    return f"Create a Spring Boot {code_type} class similar to {class_name}"


def clean_code(content: str) -> str:
    """Clean and normalise Java source code for dataset output.

    Removes package declarations, import statements, and collapses
    excessive blank lines.

    Args:
        content: Raw Java source text.

    Returns:
        The cleaned source text with leading/trailing whitespace stripped.
    """
    # Remove package declarations
    content = re.sub(r"^package\s+[\w.]+;\s*\n", "", content)
    # Remove most imports (keep essential ones in output)
    content = re.sub(r"^import\s+[\w.*]+;\s*\n", "", content, flags=re.MULTILINE)
    # Remove excessive blank lines
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def process_java_file(
    filepath: Path,
    min_examples: int = 200,
    max_length: int = 15000,
) -> Optional[Dict]:
    """Process a single Java file and create a training example.

    Reads the file, detects the Spring Boot component type, generates an
    instruction, and returns a dict ready for JSONL serialisation.

    Args:
        filepath: Absolute or relative ``Path`` to a ``.java`` file.
        min_examples: Minimum character length for the cleaned output to
            be accepted.
        max_length: Maximum character length for the cleaned output.

    Returns:
        A dict with keys ``instruction``, ``input``, and ``output``, or
        ``None`` if the file should be skipped.
    """
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", filepath, exc)
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

    # Skip files that do not meet length constraints
    if len(output) < min_examples or len(output) > max_length:
        logger.debug(
            "Skipped %s (length %d outside [%d, %d])",
            filepath.name,
            len(output),
            min_examples,
            max_length,
        )
        return None

    return {"instruction": instruction, "input": "", "output": output}


def scan_directory(
    directory: Path,
    min_examples: int = 200,
    max_length: int = 15000,
) -> List[Dict]:
    """Scan a directory tree for Java files and generate training examples.

    Recursively finds every ``.java`` file under *directory* (excluding
    paths containing ``/test/``), processes each one, and returns a list
    of valid training examples.

    Args:
        directory: Root directory to walk.
        min_examples: Forwarded to :func:`process_java_file`.
        max_length: Forwarded to :func:`process_java_file`.

    Returns:
        A list of dicts, each with ``instruction``, ``input``, and
        ``output`` keys.
    """
    examples: List[Dict] = []
    java_files = list(directory.rglob("*.java"))

    logger.info("Found %d Java files in %s", len(java_files), directory)

    processed = 0
    for filepath in java_files:
        # Skip test directories
        if "/test/" in str(filepath):
            continue

        example = process_java_file(
            filepath, min_examples=min_examples, max_length=max_length
        )
        if example:
            examples.append(example)
            processed += 1
            if processed % 50 == 0:
                logger.info("  Progress: %d examples extracted so far ...", processed)
            logger.debug("  Processed: %s", filepath.name)

    logger.info("Finished scanning %s -- %d examples extracted", directory, len(examples))
    return examples


def clone_github_repo(repo_url: str, target_dir: Path) -> bool:
    """Clone a GitHub repository with ``--depth 1``.

    If *target_dir* already exists the clone step is skipped.

    Args:
        repo_url: Repository in ``owner/repo`` format.
        target_dir: Local directory to clone into.

    Returns:
        ``True`` if the repo is available on disk (cloned or already
        existed), ``False`` on failure.
    """
    if target_dir.exists():
        logger.info("Directory %s already exists, skipping clone", target_dir)
        return True

    try:
        cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            f"https://github.com/{repo_url}.git",
            str(target_dir),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info("Cloned %s", repo_url)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to clone %s: %s", repo_url, exc)
        return False


def generate_synthetic_examples() -> List[Dict]:
    """Generate synthetic Spring Boot training examples.

    Returns a curated list of hand-written instruction/output pairs that
    cover common Spring Boot patterns including DTOs, pagination,
    application properties, security, caching, scheduling, WebSocket,
    error handling, AOP, Kafka messaging, Docker, OpenAPI, and
    React-adjacent REST API response patterns.

    Returns:
        A list of dicts with ``instruction``, ``input``, and ``output``
        keys.
    """
    synthetic: List[Dict] = []

    examples = [
        # ------------------------------------------------------------------
        # Original examples
        # ------------------------------------------------------------------
        {
            "instruction": "Create a Spring Boot DTO class with validation for user registration",
            "input": "",
            "output": """\
@Data
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
            "output": """\
@Data
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
            "output": """\
# Application
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
        # ------------------------------------------------------------------
        # Security
        # ------------------------------------------------------------------
        {
            "instruction": "Create a Spring Security configuration with JWT authentication and role-based access",
            "input": "",
            "output": """\
@Configuration
@EnableWebSecurity
@EnableMethodSecurity
public class SecurityConfig {

    private final JwtAuthenticationFilter jwtAuthFilter;
    private final UserDetailsService userDetailsService;

    public SecurityConfig(JwtAuthenticationFilter jwtAuthFilter,
                          UserDetailsService userDetailsService) {
        this.jwtAuthFilter = jwtAuthFilter;
        this.userDetailsService = userDetailsService;
    }

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            .csrf(AbstractHttpConfigurer::disable)
            .sessionManagement(session ->
                session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/auth/**", "/api/public/**").permitAll()
                .requestMatchers("/api/admin/**").hasRole("ADMIN")
                .requestMatchers("/api/user/**").hasAnyRole("USER", "ADMIN")
                .anyRequest().authenticated())
            .authenticationProvider(authenticationProvider())
            .addFilterBefore(jwtAuthFilter, UsernamePasswordAuthenticationFilter.class);
        return http.build();
    }

    @Bean
    public AuthenticationProvider authenticationProvider() {
        DaoAuthenticationProvider provider = new DaoAuthenticationProvider();
        provider.setUserDetailsService(userDetailsService);
        provider.setPasswordEncoder(passwordEncoder());
        return provider;
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration config)
            throws Exception {
        return config.getAuthenticationManager();
    }
}""",
        },
        # ------------------------------------------------------------------
        # Scheduling
        # ------------------------------------------------------------------
        {
            "instruction": "Write a Spring Boot scheduled task service for periodic data cleanup and report generation",
            "input": "",
            "output": """\
@Service
@EnableScheduling
public class ScheduledTaskService {

    private static final Logger log = LoggerFactory.getLogger(ScheduledTaskService.class);

    private final DataCleanupRepository cleanupRepository;
    private final ReportService reportService;

    public ScheduledTaskService(DataCleanupRepository cleanupRepository,
                                ReportService reportService) {
        this.cleanupRepository = cleanupRepository;
        this.reportService = reportService;
    }

    @Scheduled(cron = "0 0 2 * * ?")
    public void performNightlyCleanup() {
        log.info("Starting nightly data cleanup ...");
        LocalDateTime threshold = LocalDateTime.now().minusDays(90);
        int deleted = cleanupRepository.deleteExpiredRecordsBefore(threshold);
        log.info("Nightly cleanup complete -- removed {} expired records", deleted);
    }

    @Scheduled(fixedRate = 60_000, initialDelay = 10_000)
    public void refreshMetricsCache() {
        log.debug("Refreshing metrics cache ...");
        reportService.recalculateDashboardMetrics();
    }

    @Scheduled(cron = "0 0 8 * * MON")
    public void generateWeeklyReport() {
        log.info("Generating weekly report ...");
        try {
            reportService.generateAndEmailWeeklyReport();
            log.info("Weekly report sent successfully");
        } catch (Exception e) {
            log.error("Failed to generate weekly report", e);
        }
    }
}""",
        },
        # ------------------------------------------------------------------
        # Caching
        # ------------------------------------------------------------------
        {
            "instruction": "Create a Spring Boot Redis caching configuration with custom TTL per cache",
            "input": "",
            "output": """\
@Configuration
@EnableCaching
public class RedisCacheConfig {

    @Value("${spring.redis.host:localhost}")
    private String redisHost;

    @Value("${spring.redis.port:6379}")
    private int redisPort;

    @Bean
    public LettuceConnectionFactory redisConnectionFactory() {
        RedisStandaloneConfiguration config = new RedisStandaloneConfiguration(redisHost, redisPort);
        return new LettuceConnectionFactory(config);
    }

    @Bean
    public RedisCacheManager cacheManager(LettuceConnectionFactory connectionFactory) {
        RedisCacheConfiguration defaultConfig = RedisCacheConfiguration.defaultCacheConfig()
                .entryTtl(Duration.ofMinutes(30))
                .serializeKeysWith(
                    RedisSerializationContext.SerializationPair.fromSerializer(new StringRedisSerializer()))
                .serializeValuesWith(
                    RedisSerializationContext.SerializationPair.fromSerializer(
                        new GenericJackson2JsonRedisSerializer()))
                .disableCachingNullValues();

        Map<String, RedisCacheConfiguration> cacheConfigs = new HashMap<>();
        cacheConfigs.put("users", defaultConfig.entryTtl(Duration.ofMinutes(60)));
        cacheConfigs.put("products", defaultConfig.entryTtl(Duration.ofHours(2)));
        cacheConfigs.put("sessions", defaultConfig.entryTtl(Duration.ofMinutes(15)));

        return RedisCacheManager.builder(connectionFactory)
                .cacheDefaults(defaultConfig)
                .withInitialCacheConfigurations(cacheConfigs)
                .transactionAware()
                .build();
    }

    @Bean
    public RedisTemplate<String, Object> redisTemplate(LettuceConnectionFactory connectionFactory) {
        RedisTemplate<String, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(connectionFactory);
        template.setKeySerializer(new StringRedisSerializer());
        template.setValueSerializer(new GenericJackson2JsonRedisSerializer());
        return template;
    }
}""",
        },
        # ------------------------------------------------------------------
        # WebSocket
        # ------------------------------------------------------------------
        {
            "instruction": "Create a Spring Boot WebSocket configuration with STOMP message broker for a chat application",
            "input": "",
            "output": """\
@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {

    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        config.enableSimpleBroker("/topic", "/queue");
        config.setApplicationDestinationPrefixes("/app");
        config.setUserDestinationPrefix("/user");
    }

    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        registry.addEndpoint("/ws")
                .setAllowedOriginPatterns("*")
                .withSockJS();
    }

    @Override
    public void configureWebSocketTransport(WebSocketTransportRegistration registration) {
        registration.setMessageSizeLimit(8192)
                    .setSendBufferSizeLimit(512 * 1024)
                    .setSendTimeLimit(20_000);
    }
}

@Controller
public class ChatController {

    private final SimpMessagingTemplate messagingTemplate;

    public ChatController(SimpMessagingTemplate messagingTemplate) {
        this.messagingTemplate = messagingTemplate;
    }

    @MessageMapping("/chat.sendMessage")
    @SendTo("/topic/public")
    public ChatMessage sendMessage(@Payload ChatMessage chatMessage) {
        return chatMessage;
    }

    @MessageMapping("/chat.addUser")
    @SendTo("/topic/public")
    public ChatMessage addUser(@Payload ChatMessage chatMessage,
                               SimpMessageHeaderAccessor headerAccessor) {
        headerAccessor.getSessionAttributes().put("username", chatMessage.getSender());
        return chatMessage;
    }

    @MessageMapping("/chat.private")
    public void sendPrivateMessage(@Payload ChatMessage chatMessage) {
        messagingTemplate.convertAndSendToUser(
                chatMessage.getRecipient(), "/queue/messages", chatMessage);
    }
}""",
        },
        # ------------------------------------------------------------------
        # Global Error Handling
        # ------------------------------------------------------------------
        {
            "instruction": "Implement a global exception handler with RFC 7807 Problem Details for a Spring Boot REST API",
            "input": "",
            "output": """\
@RestControllerAdvice
public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    @ExceptionHandler(ResourceNotFoundException.class)
    public ProblemDetail handleResourceNotFound(ResourceNotFoundException ex, WebRequest request) {
        log.warn("Resource not found: {}", ex.getMessage());
        ProblemDetail problem = ProblemDetail.forStatusAndDetail(HttpStatus.NOT_FOUND, ex.getMessage());
        problem.setTitle("Resource Not Found");
        problem.setProperty("timestamp", Instant.now());
        return problem;
    }

    @ExceptionHandler(BusinessValidationException.class)
    public ProblemDetail handleBusinessValidation(BusinessValidationException ex) {
        log.warn("Business validation failed: {}", ex.getMessage());
        ProblemDetail problem = ProblemDetail.forStatusAndDetail(HttpStatus.UNPROCESSABLE_ENTITY, ex.getMessage());
        problem.setTitle("Validation Failed");
        problem.setProperty("errors", ex.getViolations());
        problem.setProperty("timestamp", Instant.now());
        return problem;
    }

    @Override
    protected ResponseEntity<Object> handleMethodArgumentNotValid(
            MethodArgumentNotValidException ex, HttpHeaders headers,
            HttpStatusCode status, WebRequest request) {
        Map<String, String> fieldErrors = new LinkedHashMap<>();
        ex.getBindingResult().getFieldErrors().forEach(error ->
                fieldErrors.put(error.getField(), error.getDefaultMessage()));

        ProblemDetail problem = ProblemDetail.forStatusAndDetail(HttpStatus.BAD_REQUEST, "Validation failed");
        problem.setTitle("Bad Request");
        problem.setProperty("fieldErrors", fieldErrors);
        problem.setProperty("timestamp", Instant.now());
        return ResponseEntity.badRequest().body(problem);
    }

    @ExceptionHandler(AccessDeniedException.class)
    public ProblemDetail handleAccessDenied(AccessDeniedException ex) {
        log.warn("Access denied: {}", ex.getMessage());
        ProblemDetail problem = ProblemDetail.forStatusAndDetail(HttpStatus.FORBIDDEN, "Access denied");
        problem.setTitle("Forbidden");
        problem.setProperty("timestamp", Instant.now());
        return problem;
    }

    @ExceptionHandler(Exception.class)
    public ProblemDetail handleUnexpected(Exception ex) {
        log.error("Unexpected error", ex);
        ProblemDetail problem = ProblemDetail.forStatusAndDetail(
                HttpStatus.INTERNAL_SERVER_ERROR, "An unexpected error occurred");
        problem.setTitle("Internal Server Error");
        problem.setProperty("timestamp", Instant.now());
        return problem;
    }
}""",
        },
        # ------------------------------------------------------------------
        # AOP
        # ------------------------------------------------------------------
        {
            "instruction": "Create a Spring AOP aspect for logging method execution time and auditing",
            "input": "",
            "output": """\
@Aspect
@Component
public class PerformanceAndAuditAspect {

    private static final Logger log = LoggerFactory.getLogger(PerformanceAndAuditAspect.class);

    @Pointcut("within(@org.springframework.web.bind.annotation.RestController *)")
    public void controllerMethods() {}

    @Pointcut("within(@org.springframework.stereotype.Service *)")
    public void serviceMethods() {}

    @Around("controllerMethods() || serviceMethods()")
    public Object logExecutionTime(ProceedingJoinPoint joinPoint) throws Throwable {
        String methodName = joinPoint.getSignature().toShortString();
        long start = System.currentTimeMillis();
        try {
            Object result = joinPoint.proceed();
            long elapsed = System.currentTimeMillis() - start;
            if (elapsed > 500) {
                log.warn("SLOW METHOD {} executed in {} ms", methodName, elapsed);
            } else {
                log.debug("{} executed in {} ms", methodName, elapsed);
            }
            return result;
        } catch (Exception ex) {
            long elapsed = System.currentTimeMillis() - start;
            log.error("{} failed after {} ms: {}", methodName, elapsed, ex.getMessage());
            throw ex;
        }
    }

    @AfterReturning(pointcut = "controllerMethods()", returning = "result")
    public void auditControllerResponse(JoinPoint joinPoint, Object result) {
        String method = joinPoint.getSignature().toShortString();
        log.info("AUDIT -- {} returned successfully", method);
    }

    @AfterThrowing(pointcut = "serviceMethods()", throwing = "ex")
    public void auditServiceException(JoinPoint joinPoint, Exception ex) {
        String method = joinPoint.getSignature().toShortString();
        log.error("AUDIT -- {} threw {}: {}", method, ex.getClass().getSimpleName(), ex.getMessage());
    }
}""",
        },
        # ------------------------------------------------------------------
        # Kafka Messaging
        # ------------------------------------------------------------------
        {
            "instruction": "Write a Spring Boot Kafka producer and consumer service for order events",
            "input": "",
            "output": """\
@Configuration
public class KafkaConfig {

    @Value("${spring.kafka.bootstrap-servers:localhost:9092}")
    private String bootstrapServers;

    @Bean
    public ProducerFactory<String, OrderEvent> producerFactory() {
        Map<String, Object> props = new HashMap<>();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class);
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, JsonSerializer.class);
        props.put(ProducerConfig.ACKS_CONFIG, "all");
        props.put(ProducerConfig.RETRIES_CONFIG, 3);
        return new DefaultKafkaProducerFactory<>(props);
    }

    @Bean
    public KafkaTemplate<String, OrderEvent> kafkaTemplate() {
        return new KafkaTemplate<>(producerFactory());
    }

    @Bean
    public ConsumerFactory<String, OrderEvent> consumerFactory() {
        Map<String, Object> props = new HashMap<>();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ConsumerConfig.GROUP_ID_CONFIG, "order-processing-group");
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class);
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, JsonDeserializer.class);
        props.put(JsonDeserializer.TRUSTED_PACKAGES, "com.example.events");
        return new DefaultKafkaConsumerFactory<>(props);
    }

    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, OrderEvent> kafkaListenerContainerFactory() {
        ConcurrentKafkaListenerContainerFactory<String, OrderEvent> factory =
                new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(consumerFactory());
        factory.setConcurrency(3);
        return factory;
    }
}

@Service
public class OrderEventProducer {

    private static final Logger log = LoggerFactory.getLogger(OrderEventProducer.class);
    private static final String TOPIC = "order-events";

    private final KafkaTemplate<String, OrderEvent> kafkaTemplate;

    public OrderEventProducer(KafkaTemplate<String, OrderEvent> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }

    public void publishOrderCreated(OrderEvent event) {
        kafkaTemplate.send(TOPIC, event.getOrderId(), event)
                .whenComplete((result, ex) -> {
                    if (ex == null) {
                        log.info("Published order event: {} to partition {}",
                                event.getOrderId(),
                                result.getRecordMetadata().partition());
                    } else {
                        log.error("Failed to publish order event: {}", event.getOrderId(), ex);
                    }
                });
    }
}

@Service
public class OrderEventConsumer {

    private static final Logger log = LoggerFactory.getLogger(OrderEventConsumer.class);

    private final OrderProcessingService processingService;

    public OrderEventConsumer(OrderProcessingService processingService) {
        this.processingService = processingService;
    }

    @KafkaListener(topics = "order-events", groupId = "order-processing-group")
    public void handleOrderEvent(OrderEvent event) {
        log.info("Received order event: {} with status {}", event.getOrderId(), event.getStatus());
        try {
            processingService.process(event);
        } catch (Exception e) {
            log.error("Error processing order event: {}", event.getOrderId(), e);
            throw e;
        }
    }
}""",
        },
        # ------------------------------------------------------------------
        # Docker / Deployment
        # ------------------------------------------------------------------
        {
            "instruction": "Create a multi-stage Dockerfile and docker-compose.yml for a Spring Boot application with PostgreSQL and Redis",
            "input": "",
            "output": """\
# ---- Dockerfile (multi-stage) ----

FROM eclipse-temurin:21-jdk-alpine AS build
WORKDIR /workspace
COPY gradle gradle
COPY gradlew build.gradle settings.gradle ./
RUN ./gradlew dependencies --no-daemon || true
COPY src src
RUN ./gradlew bootJar --no-daemon -x test

FROM eclipse-temurin:21-jre-alpine
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
WORKDIR /app
COPY --from=build /workspace/build/libs/*.jar app.jar
RUN chown -R appuser:appgroup /app
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://localhost:8080/actuator/health || exit 1
ENTRYPOINT ["java", "-XX:+UseContainerSupport", "-XX:MaxRAMPercentage=75.0", "-jar", "app.jar"]

# ---- docker-compose.yml ----

version: "3.9"
services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      SPRING_PROFILES_ACTIVE: docker
      DB_HOST: postgres
      DB_PORT: 5432
      DB_NAME: appdb
      DB_USERNAME: appuser
      DB_PASSWORD: secret
      SPRING_REDIS_HOST: redis
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: appdb
      POSTGRES_USER: appuser
      POSTGRES_PASSWORD: secret
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U appuser -d appdb"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:""",
        },
        # ------------------------------------------------------------------
        # OpenAPI / Swagger
        # ------------------------------------------------------------------
        {
            "instruction": "Create an OpenAPI/SpringDoc configuration with grouped APIs and custom info for a Spring Boot application",
            "input": "",
            "output": """\
@Configuration
public class OpenApiConfig {

    @Bean
    public OpenAPI customOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title("My Application API")
                        .version("1.0.0")
                        .description("REST API documentation for My Application")
                        .contact(new Contact()
                                .name("API Support")
                                .email("support@example.com")
                                .url("https://example.com"))
                        .license(new License()
                                .name("Apache 2.0")
                                .url("https://www.apache.org/licenses/LICENSE-2.0")))
                .externalDocs(new ExternalDocumentation()
                        .description("Full Documentation")
                        .url("https://docs.example.com"))
                .addSecurityItem(new SecurityRequirement().addList("bearerAuth"))
                .components(new Components()
                        .addSecuritySchemes("bearerAuth",
                                new SecurityScheme()
                                        .type(SecurityScheme.Type.HTTP)
                                        .scheme("bearer")
                                        .bearerFormat("JWT")));
    }

    @Bean
    public GroupedOpenApi publicApi() {
        return GroupedOpenApi.builder()
                .group("public")
                .pathsToMatch("/api/public/**")
                .build();
    }

    @Bean
    public GroupedOpenApi adminApi() {
        return GroupedOpenApi.builder()
                .group("admin")
                .pathsToMatch("/api/admin/**")
                .build();
    }

    @Bean
    public GroupedOpenApi userApi() {
        return GroupedOpenApi.builder()
                .group("user")
                .pathsToMatch("/api/user/**", "/api/profile/**")
                .build();
    }
}""",
        },
        # ------------------------------------------------------------------
        # Custom Validation
        # ------------------------------------------------------------------
        {
            "instruction": "Create a custom Bean Validation constraint and validator for validating a strong password in Spring Boot",
            "input": "",
            "output": """\
@Target({ElementType.FIELD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
@Constraint(validatedBy = StrongPasswordValidator.class)
@Documented
public @interface StrongPassword {
    String message() default "Password does not meet strength requirements";
    Class<?>[] groups() default {};
    Class<? extends Payload>[] payload() default {};
    int minLength() default 8;
    boolean requireUppercase() default true;
    boolean requireLowercase() default true;
    boolean requireDigit() default true;
    boolean requireSpecialChar() default true;
}

public class StrongPasswordValidator implements ConstraintValidator<StrongPassword, String> {

    private int minLength;
    private boolean requireUppercase;
    private boolean requireLowercase;
    private boolean requireDigit;
    private boolean requireSpecialChar;

    @Override
    public void initialize(StrongPassword annotation) {
        this.minLength = annotation.minLength();
        this.requireUppercase = annotation.requireUppercase();
        this.requireLowercase = annotation.requireLowercase();
        this.requireDigit = annotation.requireDigit();
        this.requireSpecialChar = annotation.requireSpecialChar();
    }

    @Override
    public boolean isValid(String value, ConstraintValidatorContext context) {
        if (value == null) {
            return false;
        }

        List<String> violations = new ArrayList<>();

        if (value.length() < minLength) {
            violations.add("must be at least " + minLength + " characters");
        }
        if (requireUppercase && !value.matches(".*[A-Z].*")) {
            violations.add("must contain an uppercase letter");
        }
        if (requireLowercase && !value.matches(".*[a-z].*")) {
            violations.add("must contain a lowercase letter");
        }
        if (requireDigit && !value.matches(".*\\\\d.*")) {
            violations.add("must contain a digit");
        }
        if (requireSpecialChar && !value.matches(".*[!@#$%^&*()_+\\\\-=\\\\[\\\\]{};':\"\\\\\\\\|,.<>/?].*")) {
            violations.add("must contain a special character");
        }

        if (!violations.isEmpty()) {
            context.disableDefaultConstraintViolation();
            context.buildConstraintViolationWithTemplate(
                    "Password " + String.join(", ", violations))
                    .addConstraintViolation();
            return false;
        }
        return true;
    }
}""",
        },
        # ------------------------------------------------------------------
        # React-adjacent: REST API responses consumed by React frontends
        # ------------------------------------------------------------------
        {
            "instruction": "Create a Spring Boot REST controller that returns paginated product data for a React frontend with filtering and sorting",
            "input": "",
            "output": """\
@RestController
@RequestMapping("/api/products")
@CrossOrigin(origins = "${app.cors.allowed-origins:http://localhost:3000}")
public class ProductController {

    private final ProductService productService;

    public ProductController(ProductService productService) {
        this.productService = productService;
    }

    @GetMapping
    public ResponseEntity<PageResponse<ProductDTO>> listProducts(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(defaultValue = "createdAt") String sortBy,
            @RequestParam(defaultValue = "desc") String sortDir,
            @RequestParam(required = false) String category,
            @RequestParam(required = false) BigDecimal minPrice,
            @RequestParam(required = false) BigDecimal maxPrice,
            @RequestParam(required = false) String search) {

        Sort sort = sortDir.equalsIgnoreCase("asc")
                ? Sort.by(sortBy).ascending()
                : Sort.by(sortBy).descending();
        Pageable pageable = PageRequest.of(page, size, sort);

        ProductFilter filter = ProductFilter.builder()
                .category(category)
                .minPrice(minPrice)
                .maxPrice(maxPrice)
                .searchTerm(search)
                .build();

        Page<ProductDTO> result = productService.findProducts(filter, pageable);
        return ResponseEntity.ok(PageResponse.from(result));
    }

    @GetMapping("/{id}")
    public ResponseEntity<ProductDTO> getProduct(@PathVariable Long id) {
        return ResponseEntity.ok(productService.findById(id));
    }

    @PostMapping
    public ResponseEntity<ProductDTO> createProduct(@Valid @RequestBody CreateProductRequest request) {
        ProductDTO created = productService.create(request);
        URI location = ServletUriComponentsBuilder.fromCurrentRequest()
                .path("/{id}").buildAndExpand(created.getId()).toUri();
        return ResponseEntity.created(location).body(created);
    }

    @PutMapping("/{id}")
    public ResponseEntity<ProductDTO> updateProduct(
            @PathVariable Long id,
            @Valid @RequestBody UpdateProductRequest request) {
        return ResponseEntity.ok(productService.update(id, request));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteProduct(@PathVariable Long id) {
        productService.delete(id);
        return ResponseEntity.noContent().build();
    }
}""",
        },
        {
            "instruction": "Write a Spring Boot REST endpoint that returns a unified JSON response envelope for a React SPA with loading states and error details",
            "input": "",
            "output": """\
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ApiResponse<T> {

    private boolean success;
    private T data;
    private String message;
    private List<ApiError> errors;
    private Map<String, Object> meta;

    public static <T> ApiResponse<T> ok(T data) {
        return ApiResponse.<T>builder()
                .success(true)
                .data(data)
                .build();
    }

    public static <T> ApiResponse<T> ok(T data, String message) {
        return ApiResponse.<T>builder()
                .success(true)
                .data(data)
                .message(message)
                .build();
    }

    public static <T> ApiResponse<T> error(String message, List<ApiError> errors) {
        return ApiResponse.<T>builder()
                .success(false)
                .message(message)
                .errors(errors)
                .build();
    }

    public static <T> ApiResponse<T> withMeta(T data, Map<String, Object> meta) {
        return ApiResponse.<T>builder()
                .success(true)
                .data(data)
                .meta(meta)
                .build();
    }
}

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ApiError {
    private String field;
    private String code;
    private String message;
}

@RestController
@RequestMapping("/api/v1/users")
@CrossOrigin(origins = "${app.cors.allowed-origins:http://localhost:3000}")
public class UserApiController {

    private final UserService userService;

    public UserApiController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping
    public ResponseEntity<ApiResponse<List<UserDTO>>> listUsers(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "25") int size) {

        Page<UserDTO> result = userService.findAll(PageRequest.of(page, size));
        Map<String, Object> meta = Map.of(
                "page", result.getNumber(),
                "totalPages", result.getTotalPages(),
                "totalElements", result.getTotalElements(),
                "hasNext", result.hasNext());
        return ResponseEntity.ok(ApiResponse.withMeta(result.getContent(), meta));
    }

    @PostMapping
    public ResponseEntity<ApiResponse<UserDTO>> createUser(@Valid @RequestBody CreateUserRequest request) {
        UserDTO user = userService.create(request);
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(ApiResponse.ok(user, "User created successfully"));
    }
}""",
        },
        {
            "instruction": "Create a Spring Boot SSE (Server-Sent Events) endpoint for real-time notifications consumed by a React frontend",
            "input": "",
            "output": """\
@RestController
@RequestMapping("/api/notifications")
@CrossOrigin(origins = "${app.cors.allowed-origins:http://localhost:3000}")
public class NotificationController {

    private final NotificationService notificationService;
    private final Map<String, SseEmitter> emitters = new ConcurrentHashMap<>();

    public NotificationController(NotificationService notificationService) {
        this.notificationService = notificationService;
    }

    @GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamNotifications(@RequestParam String userId) {
        SseEmitter emitter = new SseEmitter(Long.MAX_VALUE);

        emitters.put(userId, emitter);

        emitter.onCompletion(() -> emitters.remove(userId));
        emitter.onTimeout(() -> emitters.remove(userId));
        emitter.onError(e -> emitters.remove(userId));

        // Send initial connection event
        try {
            emitter.send(SseEmitter.event()
                    .name("connected")
                    .data(Map.of("status", "connected", "userId", userId)));
        } catch (IOException e) {
            emitter.completeWithError(e);
        }

        return emitter;
    }

    public void sendNotification(String userId, NotificationDTO notification) {
        SseEmitter emitter = emitters.get(userId);
        if (emitter != null) {
            try {
                emitter.send(SseEmitter.event()
                        .name("notification")
                        .data(notification, MediaType.APPLICATION_JSON));
            } catch (IOException e) {
                emitters.remove(userId);
            }
        }
    }

    public void broadcast(NotificationDTO notification) {
        List<String> deadEmitters = new ArrayList<>();
        emitters.forEach((userId, emitter) -> {
            try {
                emitter.send(SseEmitter.event()
                        .name("broadcast")
                        .data(notification, MediaType.APPLICATION_JSON));
            } catch (IOException e) {
                deadEmitters.add(userId);
            }
        });
        deadEmitters.forEach(emitters::remove);
    }

    @GetMapping
    public ResponseEntity<List<NotificationDTO>> getUnread(@RequestParam String userId) {
        return ResponseEntity.ok(notificationService.getUnreadNotifications(userId));
    }

    @PatchMapping("/{id}/read")
    public ResponseEntity<Void> markAsRead(@PathVariable Long id) {
        notificationService.markAsRead(id);
        return ResponseEntity.noContent().build();
    }
}""",
        },
        {
            "instruction": "Write a Spring Boot REST controller for file upload/download that serves assets to a React application",
            "input": "",
            "output": """\
@RestController
@RequestMapping("/api/files")
@CrossOrigin(origins = "${app.cors.allowed-origins:http://localhost:3000}")
public class FileController {

    private static final Logger log = LoggerFactory.getLogger(FileController.class);

    private final FileStorageService storageService;

    @Value("${app.upload.max-size:10485760}")
    private long maxFileSize;

    public FileController(FileStorageService storageService) {
        this.storageService = storageService;
    }

    @PostMapping("/upload")
    public ResponseEntity<FileUploadResponse> uploadFile(
            @RequestParam("file") MultipartFile file,
            @RequestParam(required = false) String folder) {

        if (file.isEmpty()) {
            throw new BadRequestException("File must not be empty");
        }
        if (file.getSize() > maxFileSize) {
            throw new BadRequestException("File size exceeds maximum allowed size");
        }

        String contentType = file.getContentType();
        if (contentType == null || !isAllowedContentType(contentType)) {
            throw new BadRequestException("File type not allowed: " + contentType);
        }

        FileMetadata metadata = storageService.store(file, folder);
        log.info("File uploaded: {} ({} bytes)", metadata.getFilename(), metadata.getSize());

        FileUploadResponse response = FileUploadResponse.builder()
                .id(metadata.getId())
                .filename(metadata.getFilename())
                .url("/api/files/" + metadata.getId())
                .contentType(metadata.getContentType())
                .size(metadata.getSize())
                .uploadedAt(metadata.getUploadedAt())
                .build();

        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @PostMapping("/upload/batch")
    public ResponseEntity<List<FileUploadResponse>> uploadMultiple(
            @RequestParam("files") List<MultipartFile> files) {
        List<FileUploadResponse> responses = files.stream()
                .map(file -> uploadFile(file, null).getBody())
                .collect(Collectors.toList());
        return ResponseEntity.status(HttpStatus.CREATED).body(responses);
    }

    @GetMapping("/{fileId}")
    public ResponseEntity<Resource> downloadFile(@PathVariable String fileId) {
        FileMetadata metadata = storageService.getMetadata(fileId);
        Resource resource = storageService.loadAsResource(fileId);

        return ResponseEntity.ok()
                .contentType(MediaType.parseMediaType(metadata.getContentType()))
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "attachment; filename=\"" + metadata.getFilename() + "\"")
                .body(resource);
    }

    @DeleteMapping("/{fileId}")
    public ResponseEntity<Void> deleteFile(@PathVariable String fileId) {
        storageService.delete(fileId);
        return ResponseEntity.noContent().build();
    }

    private boolean isAllowedContentType(String contentType) {
        return contentType.startsWith("image/")
                || contentType.equals("application/pdf")
                || contentType.equals("text/csv")
                || contentType.equals("application/json");
    }
}""",
        },
        {
            "instruction": "Create a Spring Boot REST controller for user authentication that returns JWT tokens for a React SPA",
            "input": "",
            "output": """\
@RestController
@RequestMapping("/api/auth")
@CrossOrigin(origins = "${app.cors.allowed-origins:http://localhost:3000}")
public class AuthController {

    private final AuthenticationManager authenticationManager;
    private final JwtTokenProvider tokenProvider;
    private final UserService userService;
    private final RefreshTokenService refreshTokenService;

    public AuthController(AuthenticationManager authenticationManager,
                          JwtTokenProvider tokenProvider,
                          UserService userService,
                          RefreshTokenService refreshTokenService) {
        this.authenticationManager = authenticationManager;
        this.tokenProvider = tokenProvider;
        this.userService = userService;
        this.refreshTokenService = refreshTokenService;
    }

    @PostMapping("/login")
    public ResponseEntity<AuthResponse> login(@Valid @RequestBody LoginRequest request) {
        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(request.getEmail(), request.getPassword()));

        SecurityContextHolder.getContext().setAuthentication(authentication);

        UserPrincipal principal = (UserPrincipal) authentication.getPrincipal();
        String accessToken = tokenProvider.generateAccessToken(principal);
        String refreshToken = refreshTokenService.createRefreshToken(principal.getId()).getToken();

        return ResponseEntity.ok(AuthResponse.builder()
                .accessToken(accessToken)
                .refreshToken(refreshToken)
                .tokenType("Bearer")
                .expiresIn(tokenProvider.getAccessTokenExpiration())
                .user(UserDTO.from(principal))
                .build());
    }

    @PostMapping("/register")
    public ResponseEntity<AuthResponse> register(@Valid @RequestBody RegisterRequest request) {
        if (userService.existsByEmail(request.getEmail())) {
            throw new ConflictException("Email is already in use");
        }

        UserDTO user = userService.register(request);
        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(request.getEmail(), request.getPassword()));

        UserPrincipal principal = (UserPrincipal) authentication.getPrincipal();
        String accessToken = tokenProvider.generateAccessToken(principal);
        String refreshToken = refreshTokenService.createRefreshToken(principal.getId()).getToken();

        return ResponseEntity.status(HttpStatus.CREATED).body(AuthResponse.builder()
                .accessToken(accessToken)
                .refreshToken(refreshToken)
                .tokenType("Bearer")
                .expiresIn(tokenProvider.getAccessTokenExpiration())
                .user(user)
                .build());
    }

    @PostMapping("/refresh")
    public ResponseEntity<AuthResponse> refreshToken(@Valid @RequestBody RefreshTokenRequest request) {
        return refreshTokenService.findByToken(request.getRefreshToken())
                .map(refreshTokenService::verifyExpiration)
                .map(RefreshToken::getUser)
                .map(user -> {
                    String accessToken = tokenProvider.generateAccessTokenForUser(user);
                    return ResponseEntity.ok(AuthResponse.builder()
                            .accessToken(accessToken)
                            .refreshToken(request.getRefreshToken())
                            .tokenType("Bearer")
                            .expiresIn(tokenProvider.getAccessTokenExpiration())
                            .user(UserDTO.from(user))
                            .build());
                })
                .orElseThrow(() -> new UnauthorizedException("Invalid refresh token"));
    }

    @PostMapping("/logout")
    public ResponseEntity<Void> logout(@RequestHeader("Authorization") String authHeader) {
        if (authHeader != null && authHeader.startsWith("Bearer ")) {
            String token = authHeader.substring(7);
            tokenProvider.invalidateToken(token);
        }
        return ResponseEntity.noContent().build();
    }

    @GetMapping("/me")
    public ResponseEntity<UserDTO> getCurrentUser(@AuthenticationPrincipal UserPrincipal principal) {
        return ResponseEntity.ok(UserDTO.from(principal));
    }
}""",
        },
    ]

    synthetic.extend(examples)
    return synthetic


def main() -> None:
    """Entry point for the Spring Boot dataset generator CLI.

    Parses command-line arguments, clones GitHub repositories (or uses
    defaults), scans for Java source files, optionally adds synthetic
    examples, and writes the resulting JSONL dataset to disk.
    """
    parser = argparse.ArgumentParser(
        description="Generate Spring Boot training dataset"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="datasets/spring-boot-extended.jsonl",
        help="Output file path (default: datasets/spring-boot-extended.jsonl)",
    )
    parser.add_argument(
        "--github-repo",
        "-g",
        action="append",
        help="GitHub repo to process (owner/repo format, may be repeated)",
    )
    parser.add_argument(
        "--local-dir",
        "-d",
        help="Local directory to scan for Java files",
    )
    parser.add_argument(
        "--add-synthetic",
        action="store_true",
        help="Include hand-written synthetic examples in the output",
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=200,
        help="Minimum character length for a code example to be included (default: 200)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=15000,
        help="Maximum character length for a code example to be included (default: 15000)",
    )

    args = parser.parse_args()

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    all_examples: List[Dict] = []

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
            examples = scan_directory(
                repo_dir,
                min_examples=args.min_examples,
                max_length=args.max_length,
            )
            all_examples.extend(examples)
            logger.info("Extracted %d examples from %s", len(examples), repo)

    # Process local directory if specified
    if args.local_dir:
        local_path = Path(args.local_dir)
        if local_path.exists():
            examples = scan_directory(
                local_path,
                min_examples=args.min_examples,
                max_length=args.max_length,
            )
            all_examples.extend(examples)
            logger.info("Extracted %d examples from local directory", len(examples))
        else:
            logger.error("Local directory does not exist: %s", args.local_dir)

    # Add synthetic examples
    if args.add_synthetic:
        synthetic = generate_synthetic_examples()
        all_examples.extend(synthetic)
        logger.info("Added %d synthetic examples", len(synthetic))

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for example in all_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Failed to write output file %s: %s", output_path, exc)
        raise SystemExit(1) from exc

    logger.info("Generated %d training examples", len(all_examples))
    logger.info("Output saved to: %s", output_path)


if __name__ == "__main__":
    main()
