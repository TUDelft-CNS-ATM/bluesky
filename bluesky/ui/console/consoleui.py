import string
from typing import Union
from textual import events
from textual.app import App
from textual.keys import Keys
from textual.widgets import Placeholder, ScrollView
from textual.widget import Widget
from textual.reactive import Reactive
from textual.views import DockView
from rich import box
from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.style import Style

import bluesky as bs
from bluesky.network.client import Client
from bluesky.tools.misc import tim2txt


class ConsoleClient(Client):
    '''
        Subclassed Client with a timer to periodically check for incoming data,
        an overridden event function to handle data, and a stack function to
        send stack commands to BlueSky.
    '''
    modes = ['Init', 'Hold', 'Operate', 'End']

    def __init__(self, actnode_topics=b''):
        super().__init__(actnode_topics)
        self.subscribe(b'SIMINFO')

    def event(self, name, data, sender_id):
        ''' Overridden event function. '''
        pass

    def echo(self, text, flags=None, sender_id=None):
        ''' Overload Client's echo function. '''
        if ConsoleUI.instance is not None:
            ConsoleUI.instance.echo(text, flags)

    def actnode_changed(self, newact):
        pass

    def stream(self, name, data, sender_id):
        if name == b'SIMINFO' and ConsoleUI.instance is not None:
            speed, simdt, simt, simutc, ntraf, state, scenname = data
            simt = tim2txt(simt)[:-3]
            # self.setNodeInfo(sender_id, simt, scenname)
            if sender_id == bs.net.actnode():
                ConsoleUI.instance.set_infoline(f'[b]t:[/b] {simt} [b]dt:[/b] {simdt} [b]Speed:[/b] {speed:.1f} [b]UTC:[/b] {simutc} [b]Mode:[/b] {self.modes[state]} [b]Aircraft[/b] {ntraf}')
                # acdata = bs.net.get_nodedata().acdata
                # self.siminfoLabel.setText(u'<b>t:</b> %s, <b>\u0394t:</b> %.2f, <b>Speed:</b> %.1fx, <b>UTC:</b> %s, <b>Mode:</b> %s, <b>Aircraft:</b> %d, <b>Conflicts:</b> %d/%d, <b>LoS:</b> %d/%d'
                #                           % (simt, simdt, speed, simutc, self.modes[state], ntraf, acdata.nconf_cur, acdata.nconf_tot, acdata.nlos_cur, acdata.nlos_tot))

class Echobox(ScrollView):
    text: Union[Reactive[str], str] = Reactive("")

    async def watch_text(self, text) -> None:
        await self.update(Panel(Text(text), height=max(8, 2 + text.count('\n')), box=box.SIMPLE, style=Style(bgcolor="grey53")))

    def set_text(self, text: str) -> None:
        self.text = text


class Textline(Widget):
    text: Union[Reactive[str], str] = Reactive("")
    style = Style(bgcolor="grey37")

    def __init__(self, text: str = "", name: str | None = None) -> None:
        super().__init__(name)
        self.text = text

    def render(self) -> RenderableType:
        return self.text

    def set_text(self, text: str) -> None:
        self.text = text


class ConsoleUI(App):
    cmdtext: Union[Reactive[str], str] = Reactive("")
    echotext: Union[Reactive[str], str] = Reactive("")
    infotext: Union[Reactive[str], str] = Reactive("")

    cmdbox: Textline
    echobox: Echobox
    infoline: Textline
    instance: App

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ConsoleUI.instance = self

    def echo(self, text, flags=None):
        # if flags != bs.BS_OK:
        #     text = f'[red]{text}[/red]'
        self.echotext = text + '\n' + self.echotext

    def set_infoline(self, text):
        self.infotext = text

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

    async def watch_infotext(self, infotext) -> None:
        self.infoline.set_text(f"[black]Current node:[/black] {infotext}")

    async def watch_cmdtext(self, cmdtext) -> None:
        self.cmdbox.set_text(f"[blue]>>[/blue] {cmdtext}")

    async def watch_echotext(self, echotext) -> None:
        self.echobox.set_text(echotext)

    async def on_mount(self, event: events.Mount) -> None:
        self.cmdbox = Textline("[blue]>>[/blue]")
        self.echobox = Echobox(Panel(Text(), height=8, box=box.SIMPLE, style=Style(bgcolor="grey53")))
        self.infoline = Textline("[black]Current node: [/black]")

        await self.view.dock(self.cmdbox, edge="bottom", size=1)
        # await self.view.dock(self.echobox, edge="bottom", size=8)
        echorow = DockView()
        await echorow.dock(Placeholder(), edge="right", size=20)
        await echorow.dock(self.echobox, edge="left")
        
        await self.view.dock(echorow, edge="bottom", size=8)
        # await self.view.dock(Placeholder(), edge="right", size=20)
        await self.view.dock(self.infoline, edge="bottom", size=1)

        await self.view.dock(Placeholder(), edge="top")
        await self.set_focus(self.cmdbox)

        self.set_interval(0.2, bs.net.update, name='Network')


def start():
    bs.init(mode="client")

    # Create and start BlueSky client
    bsclient = ConsoleClient()
    bsclient.connect(event_port=11000, stream_port=11001)

    ConsoleUI.run(log="textual.log")
