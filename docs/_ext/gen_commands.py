""" Generate MyST Markdown pages for the BlueSky stack command reference.

This module is imported by docs/conf.py at build time so the command
reference is always regenerated from the actual code on every build (nothing
under reference/commands/generated/ is committed to git). It can also be run
standalone for local testing:

    python docs/_ext/gen_commands.py
"""
import re
import sys
import tempfile
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent
DEEPDIVE_DIR = DOCS_DIR / 'reference' / 'commands'

# Ordered module-prefix -> category mapping. First match wins.
CATEGORIES = [
    ('bluesky.traffic.traffic', 'Aircraft creation & state'),
    ('bluesky.traffic.autopilot', 'Autopilot & guidance'),
    ('bluesky.traffic.activewpdata', 'Autopilot & guidance'),
    ('bluesky.traffic.route', 'Route & FMS'),
    ('bluesky.traffic.asas', 'Conflict detection & resolution'),
    ('bluesky.traffic.performance', 'Performance models'),
    ('bluesky.traffic.windsim', 'Wind & weather'),
    ('bluesky.traffic.turbulence', 'Wind & weather'),
    ('bluesky.traffic.trails', 'Surveillance & trails'),
    ('bluesky.traffic.adsbmodel', 'Surveillance & trails'),
    ('bluesky.traffic.aporasas', 'Surveillance & trails'),
    ('bluesky.traffic.conditional', 'Simulation & scenario control'),
    ('bluesky.traffic.trafficgroups', 'Aircraft creation & state'),
    ('bluesky.traffic.metric', 'Logging & plotting'),
    ('bluesky.simulation', 'Simulation & scenario control'),
    ('bluesky.stack', 'Simulation & scenario control'),
    ('bluesky.tools.areafilter', 'Areas & shapes'),
    ('bluesky.tools.datalog', 'Logging & plotting'),
    ('bluesky.tools.plotter', 'Logging & plotting'),
    ('bluesky.tools.datafeed', 'Surveillance & trails'),
    ('bluesky.core.plugin', 'Plugins'),
    ('bluesky.core', 'Inspection & implementations'),
    ('bluesky.network', 'Nodes & networking'),
    ('bluesky.ui', 'Display & GUI'),
    ('bluesky.plugins', 'Plugin commands'),
]

# Explicit overrides for commands that don't categorize cleanly by module.
OVERRIDES = {
    'CALC': 'Miscellaneous',
    'TMX': 'Miscellaneous',
}


def _category_for(cmd):
    name = getattr(cmd, 'name', '')
    if name in OVERRIDES:
        return OVERRIDES[name]
    module = getattr(cmd.callback, '__module__', '') or ''
    for prefix, category in CATEGORIES:
        if module == prefix or module.startswith(prefix + '.'):
            return category
    return 'Miscellaneous'


def _slug(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def _has_deepdive(name):
    return (DEEPDIVE_DIR / f'{name.lower()}.md').is_file()


def _escape_md(text):
    return (text or '').replace('|', r'\|').replace('\n', ' ').strip()


def collect_commands():
    """ Initialize BlueSky in detached mode and return the unique set of
        registered stack Command objects. """
    import bluesky as bs
    from bluesky.stack.cmdparser import Command

    workdir = tempfile.mkdtemp(prefix='bluesky-docs-')
    bs.init(mode='sim', detached=True, workdir=workdir)
    return sorted(set(Command.cmddict.values()), key=lambda c: c.name)


def _command_entry_md(cmd):
    lines = [f'## {cmd.name}', '']
    if cmd.aliases:
        lines.append(f'*Aliases: {", ".join(cmd.aliases)}*')
        lines.append('')
    lines.append('```text')
    lines.append(cmd.brief or cmd.name)
    lines.append('```')
    lines.append('')
    if cmd.help:
        lines.append(cmd.help)
        lines.append('')
    if cmd.params:
        lines.append('| Argument | Optional |')
        lines.append('|----------|----------|')
        for p in cmd.params:
            lines.append(f'| {_escape_md(str(p))} | {p.hasdefault()} |')
        lines.append('')
    else:
        lines.append('This command takes no arguments.')
        lines.append('')
    if _has_deepdive(cmd.name):
        lines.append(f'See the full [{cmd.name} guide](../{cmd.name.lower()}.md) for a worked example.')
        lines.append('')
    return '\n'.join(lines)


def generate_command_pages(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    commands = collect_commands()

    by_category = {}
    for cmd in commands:
        by_category.setdefault(_category_for(cmd), []).append(cmd)

    misc_count = len(by_category.get('Miscellaneous', []))
    if misc_count >= 20:
        raise RuntimeError(
            f"{misc_count} stack commands fell into the 'Miscellaneous' category "
            "(threshold 20) - the module->category map in gen_commands.py is out "
            "of date and needs a new entry."
        )

    category_order = sorted(by_category, key=lambda c: (c == 'Miscellaneous', c))

    # Alphabetical index-table page
    index_lines = [
        '# All stack commands', '',
        f'BlueSky currently registers **{len(commands)}** stack commands. '
        'This table lists all of them alphabetically; see the category pages '
        'in the sidebar for full descriptions and arguments.', '',
        '| Command | Usage | Description | Category |',
        '|---------|-------|--------------|----------|',
    ]
    for cmd in commands:
        cat = _category_for(cmd)
        anchor = f'{_slug(cat)}.md#{cmd.name.lower()}'
        brief = _escape_md(cmd.brief)
        desc = _escape_md(cmd.help.splitlines()[0] if cmd.help else '')
        index_lines.append(f'| [{cmd.name}]({anchor}) | `{brief}` | {desc} | {cat} |')
    (outdir / 'index-table.md').write_text('\n'.join(index_lines) + '\n')

    # One page per category
    for cat in category_order:
        cmds = sorted(by_category[cat], key=lambda c: c.name)
        lines = [f'# {cat}', '']
        for cmd in cmds:
            lines.append(_command_entry_md(cmd))
        (outdir / f'{_slug(cat)}.md').write_text('\n'.join(lines) + '\n')

    return commands, category_order


if __name__ == '__main__':
    outdir = DOCS_DIR / 'reference' / 'commands' / 'generated'
    cmds, cats = generate_command_pages(outdir)
    print(f'Generated {len(cmds)} commands across {len(cats)} categories into {outdir}')
    for cat in cats:
        n = sum(1 for c in cmds if _category_for(c) == cat)
        print(f'  {cat}: {n}')
    sys.exit(0)
