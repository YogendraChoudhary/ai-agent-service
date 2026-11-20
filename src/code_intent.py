"""Fail-safe code intent: explicit requests first, ambiguous mentions left to the classifier."""
import re

_PREFIX = r"^[\s\W]*(?:(?:please|pls|kindly|can you|could you|would you|help me|i want you to|i need you to)\s+)*"
_LANG = r'(?:python|javascript|typescript|js|ts|react|sql|bash|shell|html|css|java|rust|go|c\+\+|c#|ruby|php|swift|kotlin)'
_NOUN = rf'(?:code|function|script|component|class|program|query|regex|regular expression|endpoint|command|{_LANG})'
# "code"/"function" as a programming request, not "a code name", "a code word", "a function plan", "a function invitation",
# "a code for the treasure hunt riddle" (live audit 2026-11-20: all went to the code writer).
_NOT_PROGRAMMING = (r'(?!\s+(?:names?|words?|of\s+conduct|plans?|invitations?|events?|party|parties|rooms?|halls?|venues?|menus?|'
                    r'dress|attire|for\s+(?:the\s+|a\s+|our\s+|my\s+)?(?:treasure|hunt|riddle|game|escape|door|lock|puzzle|party|event|'
                    r'wedding|reception|gala|dinner)))')
_EXPLICIT = rf'(?:code{_NOT_PROGRAMMING}|function{_NOT_PROGRAMMING}|regex|regular expression|endpoint|html|css|{_LANG}\s+(?:code|function|script|component|class|program|query|command|stylesheet|module|app|application|test|service))'
_BUILD = re.compile(_PREFIX + rf'(?:write|create|generate|build|implement|make)\s+(?:(?:me|us)\s+)?(?:(?:a|an|the|simple|small|new)\s+)*{_EXPLICIT}(?=\W|$)', re.I)
_DIRECT = re.compile(_PREFIX + rf'(?:fix\s+(?:this|my|the)\s+code|convert\s+.+?\s+to\s+{_LANG}\b|(?:sql query|regex|bash command)\s+(?:to|for)\b)', re.I)
_MIXED = re.compile(
    _PREFIX + r'(?:(?:write|create|generate|build|draft|explain|make)\s+)?(?:(?:me|us)\s+)?'
    r'(?:(?:a|an|the|simple|small|new|detailed|short|long|technical)\s+)*'
    r'(?:blog(?: post)?|article|tutorial|guide|essay|documentation|report|newsletter|email|code of conduct|'
    rf'{_NOUN}(?:\s+{_NOUN})?\s+(?:review|specification|explanation|documentation|tutorial|guide|article|report|summary))\b', re.I)
_CUE = re.compile(rf'\b(?:{_NOUN}|api|debug|refactor|compiler|databases?|algorithms?)\b', re.I)
_ACTION = re.compile(r'\b(?:fix|refactor|optimi[sz]e|add types to|debug)\b', re.I)


def detect_code_intent(text):
    text = text or ''
    if _MIXED.search(text):
        return {'kind': 'other', 'rule': 'mixed_content'}
    if _BUILD.search(text) or _DIRECT.search(text):
        return {'kind': 'code', 'rule': 'anchored_code'}
    if '```' in text and _ACTION.search(text.split('```', 1)[0]):
        return {'kind': 'code', 'rule': 'fenced_edit'}
    if _CUE.search(text):
        return {'kind': 'unsure', 'rule': 'code_mention'}
    return {'kind': 'other', 'rule': 'no_code_cue'}
