import string
from typing import Union
from textual import events
from textual.app import App
from textual.keys import Keys
from textual.widgets import Placeholder, ScrollView
from textual.widget import Widget
from textual.reactive import Reactive
from rich.box import Box
from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.style import Style

import bluesky as bs
from bluesky.network.client import Client


VERT: Box = Box(
    """\
│ ││
│ ││
│ ││
│ ││
│ ││
│ ││
│ ││
│ ││
"""
)


class ConsoleClient(Client):
    '''
        Subclassed Client with a timer to periodically check for incoming data,
        an overridden event function to handle data, and a stack function to
        send stack commands to BlueSky.
    '''

    def event(self, name, data, sender_id):
        ''' Overridden event function. '''
        pass

    def echo(self, text, flags=None, sender_id=None):
        ''' Overload Client's echo function. '''
        if ConsoleUI.instance is not None:
            ConsoleUI.instance.echo(text, flags)


class Cmdbox(Widget):
    text: Union[Reactive[str], str] = Reactive("")

    def render(self) -> RenderableType:
        return Panel(f"[blue]>>[/blue] {self.text}")

    def set_text(self, text: str) -> None:
        self.text = text


class Echobox(ScrollView):
    text: Union[Reactive[str], str] = Reactive("")

    async def watch_text(self, text) -> None:
        await self.update(Panel(Text(text), height=max(8, 2 + text.count('\n')), box=VERT))

    def set_text(self, text: str) -> None:
        self.text = text


class ConsoleUI(App):
    cmdtext: Union[Reactive[str], str] = Reactive("")
    echotext: Union[Reactive[str], str] = Reactive("")

    cmdbox: Cmdbox
    echobox: Echobox
    instance: App

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ConsoleUI.instance = self

    def echo(self, text, flags=None):
        self.echotext = text + '\n' + self.echotext

    async def on_key(self, key: events.Key) -> None:
        if key.key == Keys.ControlH:
            self.cmdtext = self.cmdtext[:-1]
        elif key.key == Keys.Delete:
            self.cmdtext = ""
        elif key.key == Keys.Enter:
            self.echotext = self.cmdtext + '\n' + self.echotext
            bs.stack.stack(self.cmdtext)
            self.cmdtext = ""
        elif key.key == Keys.Escape:
            await self.action_quit()
        elif key.key in string.printable:
            self.cmdtext += key.key

    async def watch_cmdtext(self, cmdtext) -> None:
        self.cmdbox.set_text(cmdtext)

    async def watch_echotext(self, echotext) -> None:
        self.echobox.set_text(echotext)

    async def on_mount(self, event: events.Mount) -> None:
        self.cmdbox = Cmdbox()
        self.echobox = Echobox(Panel(Text(), height=8, box=VERT))

        await self.view.dock(self.cmdbox, edge="bottom", size=3)
        await self.view.dock(self.echobox, edge="bottom", size=8)

        await self.view.dock(Placeholder(), edge="top")
        await self.set_focus(self.cmdbox)

        self.set_interval(0.2, bs.net.update, name='Network')


def start():
    bs.init(mode="client")

    # Create and start BlueSky client
    bsclient = ConsoleClient()
    bsclient.connect(event_port=11000, stream_port=11001)

    ConsoleUI.run(log="textual.log")
