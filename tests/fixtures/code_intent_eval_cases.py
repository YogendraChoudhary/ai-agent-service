"""Labelled regex regression corpus: (group, query, expected kind)."""
CASES = []

for query in (
    'Write a program for our annual conference', 'Create a class schedule for primary school',
    'Write a command to inspire our team', 'Write a script for a wedding toast',
    'Write a go-to-market strategy', 'Write a Swift poem about autumn',
):
    CASES.append(('ambiguous_words', query, 'unsure'))
CASES.append(('mixed', 'Write a code of conduct for staff', 'other'))

for prefix in ('', 'Please ', 'Can you ', 'Could you '):
    for query in (
        'Write a code review for this function', 'Write a function specification for the API',
        'Generate a regex explanation for beginners', 'Create a Python function documentation page',
        'Write a React component guide', 'Write a SQL query review',
    ):
        CASES.append(('code_prose', prefix + query, 'other'))

for prefix in ('', 'Please ', 'Can you ', 'Could you '):
    for query in (
        'write a Python function to sort records', 'create a JavaScript function for dates',
        'generate a React component for a menu', 'build a SQL query for monthly totals',
        'implement a TypeScript class for a cache', 'write a bash script to list files',
        'create a CSS stylesheet for a header', 'generate HTML for a login form',
        'write a Rust program to count words', 'create a Java class to parse input',
        'generate a regex for email addresses', 'write a shell command to find files',
        'fix this code: x = 1', 'convert this to TypeScript: const x = 1;',
        'SQL query to list active users', 'regex for a postcode', 'bash command to count files',
        'refactor this:\n```python\ndef f(): return 1\n```',
        'optimize this:\n```js\nconst f = x => x;\n```',
        'add types to this:\n```js\nconst f = x => x;\n```',
    ):
        CASES.append(('code', prefix + query, 'code'))

for document in ('blog post', 'article', 'tutorial', 'guide', 'email', 'newsletter', 'report', 'documentation'):
    for topic in ('Python functions', 'React components', 'SQL queries', 'CSS with a code example',
                  'API design', 'how to write code', 'JavaScript promises', 'debugging scripts'):
        CASES.append(('mixed', f'Write a {document} about {topic}', 'other'))

for query in (
    'The Python ecosystem is growing', 'I need something in Python that sorts a list',
    'Can you help with my API endpoint', 'What does this JavaScript function do?',
    'Our code needs a review', 'This SQL query is confusing', 'Tell me about React',
    'We are hiring a CSS developer', 'How do databases work?', 'Explain this algorithm',
):
    CASES.append(('mention', query, 'unsure'))

for query in (
    'Write a birthday greeting', 'Summarize the notes', 'Make this shorter', 'Translate this to French',
    'Write a poem about autumn', 'What is the weather like?', 'Plan a team dinner',
    'Describe a sunset', 'Explain our leave policy', 'Create a table of prices',
):
    CASES.append(('content', query, 'other'))
