"""Task defaults and provider candidates. Keys are resolved only when a candidate runs."""
import json
from dataclasses import dataclass

from config import settings
from .models import ADVANCED_MODEL_IDS, BASIC_MODEL_IDS, BLOCKED_MODELS, supports_vision

TASK_DEFAULTS = {'code': 'claude', 'writing': 'openai', 'content': None, 'design_canvas': 'claude'}
FALLBACK_TREES = {'code': ['claude', 'openai', 'gemini'], 'writing': ['openai', '<requested>', 'claude'],
                  'content': ['<requested>', 'openai', 'claude'],
                  'design_canvas': ['claude', 'gemini', 'openai']}
# Long-form writing actions that default to OpenAI's advanced tier. Tone and the quick edits stay on content.
WRITING_REQUEST_TYPES = ('make_longer', 'generate_email_template', 'generate_blog')


@dataclass(frozen=True)
class Candidate:
    provider: str
    advanced: bool


class KeyResolver:
    def __init__(self, user_keys, default_keys, resolve):
        self.user_keys = {**(user_keys or {})}
        if default_keys is None:
            self.user_keys['use_own_keys'] = True
        self.default_keys = default_keys
        self.resolve = resolve

    @property
    def available_providers(self):
        return {p for p in ADVANCED_MODEL_IDS if
                (self.user_keys.get('use_own_keys') and self.user_keys.get('anthropic_api_key' if p == 'claude' else f'{p}_api_key'))
                or (self.default_keys or {}).get('anthropic_api_key' if p == 'claude' else f'{p}_api_key')}

    def __call__(self, provider):
        return self.resolve(user_keys=self.user_keys, model_provider=provider, default_keys=self.default_keys)


def resolve_route(task, model, model_source, available_providers, tier, has_images=False):
    defaults = {**TASK_DEFAULTS, **json.loads(settings.task_route_defaults)}
    trees = {**FALLBACK_TREES, **json.loads(settings.task_fallback_trees)}
    first = model if model_source == 'user' or not settings.task_routing_enabled else defaults.get(task) or model
    providers = [first]
    if settings.task_routing_enabled or settings.provider_fallback_enabled:
        providers += [model if p == '<requested>' else p for p in trees.get(task, [])]
    models = ADVANCED_MODEL_IDS if tier else BASIC_MODEL_IDS
    return [Candidate(p, tier) for p in dict.fromkeys(providers)
            if p in available_providers and models.get(p) not in BLOCKED_MODELS
            and (not has_images or supports_vision(p))]


def task_for_request(request_type, query, classify=None, execution_mode: str = "general"):
    """Explicit actions always win; the caller checks image intent before this for general chat."""
    from .code_intent import detect_code_intent
    # Mode 8: design_canvas — highest priority task classification
    if execution_mode == 'design_canvas':
        return 'design_canvas', {'intent_rule': 'design_canvas_mode', 'classifier_used': False}
    if request_type in ('generate_code', 'explain_code'):
        return 'code', {'intent_rule': 'explicit_type', 'classifier_used': False}
    if request_type in WRITING_REQUEST_TYPES:
        return 'writing', {'intent_rule': 'explicit_type', 'classifier_used': False}
    if request_type != 'general' or not (settings.code_intent_regex_enabled or settings.code_intent_shadow_mode):
        return 'content', {'intent_rule': 'typed_or_disabled', 'classifier_used': False}
    decision = detect_code_intent(query)
    attrs = {'intent_rule': decision['rule'], 'classifier_used': False}
    kind = decision['kind']
    if kind == 'unsure' and settings.code_intent_classifier_enabled and classify:
        import time
        start = time.perf_counter()
        kind = classify(query)
        attrs.update(classifier_used=True, classifier_ms=round((time.perf_counter() - start) * 1000, 1),
                     classifier_result=kind)
    attrs['intent_result'] = kind
    return ('code' if kind == 'code' and not settings.code_intent_shadow_mode else 'content'), attrs
