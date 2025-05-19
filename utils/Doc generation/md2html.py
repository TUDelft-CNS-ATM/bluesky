from subprocess import Popen, PIPE
from pathlib import Path
import re


source_path = Path('bluesky.wiki/')

# Regular expression to match Github internal references like [[name|link]]
re_gitlink2 = re.compile(r'\[\[([^|]+)\|([^]]+)\]\]')
# Regular expression to match Github internal references like [[name]]
re_gitlink1 = re.compile(r'\[\[([^]|]+)\]\]')
# Replace whitespace in file names to dashes
re_ws       = re.compile(r'\s+')


def wsrepl(matchobj):
    name = matchobj.groups(0)[0]
    link = re_ws.sub('-', matchobj.groups(0)[-1])

    return '[%s](%s.html)' % (name, link)


for file in source_path.glob('*.md'):
    with open(file) as f:
        lines   = f.read()
        lines   = re_gitlink1.sub(wsrepl, lines)
        lines   = re_gitlink2.sub(wsrepl, lines)
    file = Path(file)
    fileout = file.with_suffix(".html").name
    print(file, '->', fileout)
    title = file.stem.replace('-', ' ')
    p = Popen('pandoc -o html/' + fileout + f' --template template.html  --metadata title="{title}" --css doc.css -f gfm', stdin=PIPE, shell=True)
    p.communicate(lines.encode('charmap', errors='ignore'))





# import mistune
# from pygments import highlight
# from pygments.lexers import get_lexer_by_name
# from pygments.formatters import html


# class HighlightRenderer(mistune.Renderer):
#     def block_code(self, code, lang):
#         if not lang:
#             lang = 'C'
#             # return '\n<pre><code>%s</code></pre>\n' % \
#             #     mistune.escape(code)
#         lexer = get_lexer_by_name(lang, stripall=True)
#         formatter = html.HtmlFormatter()
#         return highlight(code, lexer, formatter)


# renderer = HighlightRenderer()
# markdown = mistune.Markdown(renderer=renderer)

# with open('Simulation-control.md') as fin:
#     mdtext = fin.read()

# with open('simcontrol.html', 'w') as fout:
#     fout.write(markdown(mdtext))
