import pytest

from bluesky.core.plugin import Plugin, plugins


def write_plugin(path, name, kind='sim'):
    path.write_text(
        'def init_plugin():\n'
        f'    return {{"plugin_name": {name!r}, "plugin_type": {kind!r}}}\n',
        encoding='utf-8',
    )


def link_to(path, target, directory=False):
    try:
        path.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        if getattr(exc, 'winerror', None) == 1314:
            pytest.skip('Creating symlinks requires additional Windows privileges')
        raise


@pytest.fixture
def discover(monkeypatch):
    monkeypatch.setattr(Plugin, 'plugins', {})
    monkeypatch.setattr(Plugin, 'plugins_ext', [])

    def find(*roots):
        monkeypatch.setattr(plugins.__spec__, 'submodule_search_locations', list(map(str, roots)))
        Plugin.find_plugins('sim')
        return {name: plugin.fullname for name, plugin in Plugin.plugins.items()}

    return find


def test_discover_linked_directories(tmp_path, discover):
    root = tmp_path / 'plugins'
    root.mkdir()
    external = tmp_path / 'external'
    external.mkdir()
    write_plugin(root / 'plain.py', 'plain')
    write_plugin(external / 'sample.py', 'linked')
    write_plugin(external / 'gui_sample.py', 'gui', kind='gui')
    link_to(root / 'alias', external, directory=True)
    link_to(root / 'other_alias', external, directory=True)
    link_to(external / 'back', root, directory=True)

    found = discover(root)
    assert found.keys() == {'PLAIN', 'LINKED'}
    assert found['PLAIN'] == 'bluesky.plugins.plain'
    assert found['LINKED'] in {
        'bluesky.plugins.alias.sample',
        'bluesky.plugins.other_alias.sample',
    }
    assert Plugin.plugins_ext == ['GUI', 'GUI']


def test_discover_linked_files_and_skip_broken_links(tmp_path, discover):
    root = tmp_path / 'plugins'
    root.mkdir()
    target = tmp_path / 'external.py'
    write_plugin(target, 'linked')
    link_to(root / 'linked.py', target)
    link_to(root / 'broken.py', tmp_path / 'missing.py')

    assert discover(root) == {'LINKED': 'bluesky.plugins.linked'}


def test_discover_multiple_roots_in_order(tmp_path, discover):
    first = tmp_path / 'first'
    first.mkdir()
    second = tmp_path / 'second'
    second.mkdir()
    write_plugin(first / 'earlier.py', 'duplicate')
    write_plugin(second / 'later.py', 'duplicate')
    write_plugin(second / '__init__.py', 'package')

    assert discover(first, second) == {
        'DUPLICATE': 'bluesky.plugins.later',
        'PACKAGE': 'bluesky.plugins.__init__',
    }
