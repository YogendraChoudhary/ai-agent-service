"""
One place for the language data and regexes the formatter shares.

gate.py, rules.py and post_processor.py used to keep their own copies, and they drifted: the gate knew "good morning" as a
greeting and the post-processor did not, "monday" lived in an acronym table, and March and August were missing. Anything
that is a list of words or a pattern used by more than one module belongs here.

Casing priority when rules apply it: the user's own dictionary first (apply_dictionary), then TECH_ACRONYMS and
PROPER_NOUNS, then days/months and languages/places.
"""

from __future__ import annotations

import calendar
import re
from typing import Dict, FrozenSet, List

# --------------------------------------------------------------------------------------------------------------------
# Casing
# --------------------------------------------------------------------------------------------------------------------
# Words that are also ordinary English: capitalised only where a month is plainly meant ("in may", "may 15").
CONTEXTUAL_MONTHS: FrozenSet[str] = frozenset({"may"})

# Generated, so no month or day can be forgotten ("march" and "august" were).
DAYS_AND_MONTHS: Dict[str, str] = {
    name.lower(): name
    for name in (*calendar.day_name, *calendar.month_name)
    if name and name.lower() not in CONTEXTUAL_MONTHS
}

LANGUAGES_AND_PLACES: Dict[str, str] = {w: w.capitalize() for w in (
    "english", "spanish", "french", "german", "italian", "chinese", "japanese", "korean", "hindi", "portuguese", "russian",
    "arabic", "france", "germany", "spain", "italy", "china", "japan", "india", "canada", "europe", "africa", "asia",
    "australia", "brazil", "mexico")}

TECH_ACRONYMS: Dict[str, str] = {
    "api": "API", "aws": "AWS", "gcp": "GCP", "gke": "GKE", "eks": "EKS", "ecs": "ECS", "s3": "S3", "rds": "RDS",
    "sqs": "SQS", "sns": "SNS", "k8s": "K8s", "grpc": "gRPC", "graphql": "GraphQL", "jwt": "JWT", "oauth": "OAuth",
    "json": "JSON", "yaml": "YAML", "cicd": "CI/CD", "cors": "CORS", "csrf": "CSRF", "xss": "XSS", "sdk": "SDK",
    "ide": "IDE", "orm": "ORM", "cdn": "CDN", "dns": "DNS", "tcp": "TCP", "udp": "UDP", "tls": "TLS", "ssl": "SSL",
    "ssh": "SSH", "cli": "CLI", "gui": "GUI", "pr": "PR", "lgtm": "LGTM", "wip": "WIP", "css": "CSS", "html": "HTML",
    "sql": "SQL", "url": "URL", "uri": "URI", "ui": "UI", "ux": "UX", "ai": "AI", "ml": "ML", "cpu": "CPU", "gpu": "GPU",
    "http": "HTTP", "https": "HTTPS", "smtp": "SMTP", "ftp": "FTP", "sftp": "SFTP", "saas": "SaaS", "paas": "PaaS",
    "iaas": "IaaS", "usp": "USP", "kpi": "KPI", "roi": "ROI", "ram": "RAM", "ssd": "SSD", "pdf": "PDF", "csv": "CSV",
    "q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4",
}
# "apis" -> "APIs", "sdks" -> "SDKs" ...
TECH_ACRONYMS.update({f"{k}s": f"{v}s" for k, v in list(TECH_ACRONYMS.items())
                      if v.isalpha() and v.isupper() and len(k) <= 4 and not k.endswith("s")})

# Names an ASR writes in lower case. Words that are also ordinary English ("bun", "linear", "remix") are left out.
PROPER_NOUNS: Dict[str, str] = {
    "github": "GitHub", "gitlab": "GitLab", "postgresql": "PostgreSQL", "postgres": "Postgres", "fastapi": "FastAPI",
    "pydantic": "Pydantic", "kubernetes": "Kubernetes", "terraform": "Terraform", "pulumi": "Pulumi", "svelte": "Svelte",
    "vue": "Vue", "nextjs": "Next.js", "deno": "Deno", "redis": "Redis", "mongodb": "MongoDB", "duckdb": "DuckDB",
    "jira": "Jira", "slack": "Slack", "javascript": "JavaScript", "typescript": "TypeScript", "python": "Python",
    "docker": "Docker", "linux": "Linux", "graphql": "GraphQL", "openai": "OpenAI", "gemini": "Gemini",
}

# Later entries win: acronyms and proper nouns beat days/months and places.
TERM_CASING: Dict[str, str] = {**LANGUAGES_AND_PLACES, **DAYS_AND_MONTHS, **PROPER_NOUNS, **TECH_ACRONYMS}

# "may" is a month only here: "in may", "since may", "may 15", "may 15th".
MAY_AS_MONTH_RE = re.compile(
    r"\b(?:in|of|on|since|until|by|during|before|after|early|late|mid|every|next|last|this)\s+may\b|\bmay(?=\s+\d{1,2}(?:st|nd|rd|th)?\b)",
    re.IGNORECASE)

# --------------------------------------------------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------------------------------------------------
WH_WORDS: FrozenSet[str] = frozenset({"how", "what", "why", "who", "whom", "whose", "which"})
# Open a question whatever follows.
AUX_QUESTION_STARTERS: FrozenSet[str] = frozenset({
    "can", "could", "does", "did", "is", "are", "was", "were", "would", "should",
    "isn't", "aren't", "wasn't", "weren't", "doesn't", "didn't", "can't", "couldn't", "wouldn't", "shouldn't"})
# Also open imperatives ("Do the dishes", "Have a good day", "Will send it"): a question only when a subject follows.
AMBIGUOUS_STARTERS: FrozenSet[str] = frozenset({
    "do", "have", "has", "had", "will", "shall", "may", "might", "must", "am", "won't", "haven't", "hasn't"})
SUBJECT_WORDS: FrozenSet[str] = frozenset({
    "i", "you", "we", "they", "he", "she", "it", "anyone", "anybody", "everyone", "everybody", "someone", "there",
    "this", "that", "these", "those"})
WHEN_WHERE: FrozenSet[str] = frozenset({"when", "where"})
QUESTION_STARTERS: FrozenSet[str] = WH_WORDS | AUX_QUESTION_STARTERS | AMBIGUOUS_STARTERS | WHEN_WHERE


def starts_question(text: str) -> bool:
    """"how do I...", "when does the pod restart", "do you know" are questions; "Do the dishes" and "When we deployed" are not."""
    words: List[str] = re.findall(r"[a-z']+", text.lower())[:2]
    if not words:
        return False
    first, second = words[0], (words[1] if len(words) > 1 else "")
    if first in WH_WORDS or first in AUX_QUESTION_STARTERS:
        return True
    if first in AMBIGUOUS_STARTERS:
        return second in SUBJECT_WORDS
    if first in WHEN_WHERE:
        return second in AUX_QUESTION_STARTERS or second in {"do", "have", "has", "had", "will", "shall"}
    return False


# --------------------------------------------------------------------------------------------------------------------
# Greeting, sign-off, spoken commands
# --------------------------------------------------------------------------------------------------------------------
GREETING_KEYWORDS = r"(?:hi|hey|hello|dear|greetings|good\s+(?:morning|afternoon|evening)|to\s+whom\s+it\s+may\s+concern)"
GREETING_DETECT_RE = re.compile(rf"\A\s*{GREETING_KEYWORDS}\b", re.IGNORECASE)
# "Hey Sarah, can we..." -> group 1 "Hey Sarah,", group 2 the first character of the sentence that follows.
GREETING_SPLIT_RE = re.compile(rf"\A({GREETING_KEYWORDS}(?:\s+[^,\n.!?]{{1,30}})?,)[ \t]+(\S)", re.IGNORECASE)

SIGN_OFF_KEYWORDS = (r"(?:thanks|thank\s+you|regards|best\s+regards|kind\s+regards|warm\s+regards|best|cheers|sincerely|"
                     r"yours\s+truly)")
SIGN_OFF_CLOSING_RE = re.compile(rf"\b{SIGN_OFF_KEYWORDS}\b", re.IGNORECASE)
# "Thanks, alex." -> group 1 the sign-off, group 2 the name. Case-insensitive: Whisper writes unfamiliar names in lower case.
# The comma is required: "that is the best idea" must not read as the sign-off "Best, idea".
SIGN_OFF_RE = re.compile(
    rf"(?:\n\n|\s+)({SIGN_OFF_KEYWORDS}),[ \t]+([a-zA-Z][\w'-]{{0,20}}(?: [a-zA-Z][\w'-]{{0,20}})?)\.?\Z", re.IGNORECASE)

# Spoken punctuation and structure. "heading" is deliberately not here: Markdown output is switched off.
SPOKEN_COMMANDS_RE = re.compile(
    r"\b(?:"
    r"new\s+paragraph|new\s+line|"
    r"question\s+mark|exclamation(?:\s+mark|\s+point)?|"
    r"comma|period|full\s+stop|colon|semicolon|"
    r"(?:open|close)\s+(?:bracket|paren|parenthesis|quote|brace)|"
    r"forward\s+slash|backslash|dash|hyphen"
    r")\b",
    re.IGNORECASE)
