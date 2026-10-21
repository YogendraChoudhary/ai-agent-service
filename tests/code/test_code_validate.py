import unittest
from src.code_validate import extract_blocks, validate_code


class TestCodeValidation(unittest.TestCase):
    def test_valid_languages(self):
        cases = {
            'python': 'def add(a, b):\n    return a + b',
            'json': '{"answer": 42}',
            'javascript': 'const f = (x) => ({x}); // }\nconst r = /[({]/g;',
            'typescript': 'const value: string = `hi ${f({x: "}"})}`;',
            'css': 'a { color: red; background: url(https://example.com/a.png); }',
            'html': '<!doctype html><html><body><img src="x"><br><p>Hello</p></body></html>',
            'sql': 'SELECT * FROM users;',
        }
        for lang, code in cases.items():
            with self.subTest(lang=lang):
                self.assertEqual(validate_code(f'```{lang}\n{code}\n```'), [])

    def test_broken_code_and_fences(self):
        for text in ('print(1)', '```\nprint(1)\n```', 'Prose\n```python\nx=1\n```',
                     '```python\ndef f():\nreturn 1\n```', '```json\n{"a":}\n```',
                     '```js\nfunction f() {\n```', '```ts\nconst a = `hi ${f(1)`;\n```',
                     '```js\nconst a = "oops;\n```', '```css\na { /* unclosed\n```',
                     '```html\n<div><p></div>\n```', '```html\n<div>\n```', '```python\n\n```'):
            with self.subTest(text=text):
                self.assertTrue(validate_code(text))

    def test_multiple_files_and_nested_templates(self):
        text = '```python\n# main.py\nx=1\n```\n```js\n// main.js\nconst s = `a ${`b ${1}`}`;\n```'
        self.assertEqual(len(extract_blocks(text)), 2)
        self.assertEqual(validate_code(text), [])

    def test_jsx_text_attributes_fragments_and_expressions(self):
        for language in ('jsx', 'tsx', 'javascript'):
            for code in (
                "export default function App() { return <div>You're signed in</div>; }",
                'const App = () => <><h1>"Hello" (welcome)</h1><Button onClick={() => alert("hi")}>Don\'t leave</Button></>;',
                'const App = () => <div>{items.map(x => <span key={x.id}>{x.name}</span>)}</div>;',
                'const App = <input title="say \'hello\'" value={value ?? ""} />;',
                'const App = <div>{/* comment */}{show ? <b>Yes</b> : <i>No</i>}</div>;',
                "const App = show && <div>You're in</div>; const Other = value || <div>You're out</div>;",
            ):
                with self.subTest(language=language, code=code):
                    self.assertEqual(validate_code(f'```{language}\n{code}\n```'), [])
        self.assertEqual(validate_code('```tsx\nconst id = <T,>(x: T): T => x;\n```'), [])
        for code in ('const App = <Select<string> value="hi" />;',
                     'const App = <Select<Array<string>> value={items}>Select one</Select>;',
                     'const App = <Select<(x: string) => string> callback={x => x} />;'):
            with self.subTest(code=code):
                self.assertEqual(validate_code(f'```tsx\n{code}\n```'), [])

    def test_malformed_jsx_is_still_rejected(self):
        for code in ('const App = <div><span></div>;', 'const App = <div>hello;',
                     'const App = <div>{foo(}</div>;', 'const App = <input title="oops />;',
                     'const App = <div title={foo(}>Hello</div>;'):
            with self.subTest(code=code):
                self.assertTrue(validate_code(f'```tsx\n{code}\n```'))

    def test_unfinished_html_tags_attributes_and_comments(self):
        for code in ('<div title="hello></div>', "<div title='hello></div>", '<div><span', '<!-- no end'):
            with self.subTest(code=code):
                self.assertTrue(validate_code(f'```html\n{code}\n```'))
        for code in ('<div>&lt;span&gt;Hello</div>', '<div title="one > two">yes</div>', '<p>Hello & bye</p>'):
            with self.subTest(code=code):
                self.assertEqual(validate_code(f'```html\n{code}\n```'), [])
