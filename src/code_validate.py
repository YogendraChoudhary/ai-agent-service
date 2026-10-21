"""Validate generated code without executing it. JS/TS/CSS checks are structural, not a compiler."""
import ast
import json
import re
from html.parser import HTMLParser

_FENCE = re.compile(r'^```([\w#+.-]+)[ \t]*\r?\n(.*?)^```[ \t]*\r?$', re.M | re.S)
_VOID = frozenset('area base br col embed hr img input link meta param source track wbr'.split())


class CodeValidationError(ValueError):
    def __init__(self, errors):
        super().__init__('The generated code did not pass validation')
        self.errors = errors


def extract_blocks(text):
    return [(m[1].lower(), m[2]) for m in _FENCE.finditer(text)]


def _balance(code, language='javascript'):
    def string(i, quote):
        i += 1
        while i < len(code):
            c = code[i]
            if c == '\\':
                i += 2
            elif c == quote:
                return i + 1
            elif c in '\r\n' and quote != '`':
                raise ValueError('Unclosed string')
            elif quote == '`' and code.startswith('${', i):
                i = scan(i + 2, '}')
            else:
                i += 1
        raise ValueError('Unclosed string or template literal')

    def jsx(i):
        opening = re.match(r'<([A-Za-z_$][\w$.:-]*)(?=[\s/<>])|<>', code[i:])
        if opening is None:
            raise ValueError('Invalid JSX tag')
        name = opening[1] or ''
        i += opening.end()
        if name and code[i:i + 1] == '<':
            depth = 1
            i += 1
            while i < len(code) and depth:
                if code[i] in "\"'`":
                    i = string(i, code[i])
                    continue
                if code.startswith('=>', i):
                    i += 2
                    continue
                if code[i] == '<':
                    depth += 1
                elif code[i] == '>':
                    depth -= 1
                i += 1
            if depth:
                raise ValueError('Unclosed JSX type arguments')
        if name:
            while i < len(code):
                if code.startswith('/>', i):
                    return i + 2
                if code[i] == '>':
                    i += 1
                    break
                if code[i] in "\"'":
                    end = code.find(code[i], i + 1)
                    if end < 0:
                        raise ValueError('Unclosed JSX attribute')
                    i = end + 1
                elif code[i] == '{':
                    i = scan(i + 1, '}')
                elif code[i] == '<':
                    raise ValueError('Unclosed JSX opening tag')
                else:
                    i += 1
            else:
                raise ValueError('Unclosed JSX opening tag')
        while i < len(code):
            if code.startswith('</', i):
                closing = re.match(r'</([A-Za-z_$][\w$.:-]*)?\s*>', code[i:])
                if closing is None or (closing[1] or '') != name:
                    raise ValueError('Mismatched JSX closing tag')
                return i + closing.end()
            if code[i] == '<':
                i = jsx(i)
            elif code[i] == '{':
                i = scan(i + 1, '}')
            elif code[i] == '}':
                raise ValueError('Unmatched } in JSX text')
            else:
                # Apostrophes, quotes and brackets in JSX text are not JavaScript syntax.
                i += 1
        raise ValueError('Unclosed JSX element')

    def scan(i, closing=None):
        previous = ''
        while i < len(code):
            c = code[i]
            if c.isspace():
                i += 1
                continue
            if code.startswith('/*', i):
                end = code.find('*/', i + 2)
                if end < 0:
                    raise ValueError('Unclosed block comment')
                i = end + 2
                continue
            if language != 'css' and code.startswith('//', i):
                end = code.find('\n', i + 2)
                i = len(code) if end < 0 else end + 1
                continue
            if c in "\"'`":
                i = string(i, c)
                previous = 'value'
                continue
            if (c == '<' and language in ('js', 'javascript', 'jsx', 'tsx')
                    and previous in ('', '=', ':', ',', '?', '&', '|', '!', 'return', 'default', '>')
                    and re.match(r'<(?:[A-Za-z_$]|>)', code[i:])
                    and not re.match(r'<[A-Za-z_$][\w$]*(?:\s*,|\s+extends\b)', code[i:])):
                i = jsx(i)
                previous = 'value'
                continue
            if c in '([{':
                i = scan(i + 1, {'(': ')', '[': ']', '{': '}'}[c])
                previous = 'value'
                continue
            if c in ')]}':
                if c != closing:
                    raise ValueError(f'Unmatched {c}')
                return i + 1
            # Brackets inside regex literals are not code structure.
            if language != 'css' and c == '/' and previous in ('', '=', '(', ':', ',', '[', '!', '?', 'return', '>'):
                i += 1
                in_class = False
                while i < len(code):
                    if code[i] == '\\':
                        i += 2
                        continue
                    if code[i] == '[':
                        in_class = True
                    elif code[i] == ']':
                        in_class = False
                    elif code[i] == '/' and not in_class:
                        break
                    elif code[i] in '\r\n':
                        raise ValueError('Unclosed regex literal')
                    i += 1
                else:
                    raise ValueError('Unclosed regex literal')
                previous = 'value'
            elif c.isalpha() or c == '_':
                match = re.match(r'[\w$]+', code[i:])
                previous = match[0]
                i += len(match[0])
                continue
            else:
                previous = c
            i += 1
        if closing:
            raise ValueError(f'Unclosed bracket, expected {closing}')
        return i
    scan(0)


class _HTMLBalance(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []

    def close(self):
        # HTMLParser silently consumes unfinished tags on close.
        pending = self.rawdata
        super().close()
        if pending.startswith('<'):
            raise ValueError('Unclosed HTML tag, attribute or comment')

    def handle_starttag(self, tag, attrs):
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in _VOID:
            return
        if not self.stack or self.stack.pop() != tag:
            raise ValueError(f'Unmatched closing tag </{tag}>')

    def handle_data(self, data):
        if self.stack and self.stack[-1] in ('script', 'style'):
            _balance(data, 'css' if self.stack[-1] == 'style' else 'javascript')


def validate_code(text):
    blocks = extract_blocks(text)
    if not blocks or _FENCE.sub('', text).strip():
        return ['format: Output only fenced code blocks, each with a language tag.']
    errors = []
    for n, (lang, code) in enumerate(blocks, 1):
        try:
            if not code.strip():
                raise ValueError('Empty code block')
            if lang in ('python', 'py'):
                ast.parse(code)
            elif lang == 'json':
                json.loads(code)
            elif lang in ('js', 'javascript', 'ts', 'typescript', 'jsx', 'tsx', 'css'):
                _balance(code, lang)
            elif lang in ('html', 'htm'):
                parser = _HTMLBalance()
                parser.feed(code)
                parser.close()
                if parser.stack:
                    raise ValueError('Unclosed HTML tags: ' + ', '.join(parser.stack))
        except (SyntaxError, ValueError, RecursionError) as e:
            errors.append(f'block {n} ({lang}): {e}')
    return errors


def repair_prompt(reply, errors):
    return ('Fix only the errors in this code. Return the complete corrected answer as tagged code fences, '
            'with no prose.\nERRORS:\n' + '\n'.join(errors) + '\nPREVIOUS ANSWER:\n' + reply)
