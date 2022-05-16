from typing import NewType, TypeVar

import string
from typing import Union
from textual import events
from textual.app import App
from textual.keys import Keys
from textual.widgets import Placeholder, ScrollView, Footer, TreeControl, TreeClick, TreeNode
from textual.widget import Widget
from textual.reactive import Reactive
from textual.views import DockView
from rich.align import Align
from rich import box
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.markdown import Markdown
from rich.style import Style
import rich.repr
import numpy as np
import copy

import bluesky as bs
from bluesky.network.client import Client
from bluesky.tools.misc import tim2txt
from bluesky.tools.aero import ft, kts, nm, fpm

NodeID = NewType("NodeID", int)
NodeDataType = TypeVar("NodeDataType")
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
        self.subscribe(b'ACDATA')
        
        self.count = 0
        self.nodes = dict()

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
            self.setNodeInfo(sender_id, simt, scenname)
            if sender_id == bs.net.actnode():
                ConsoleUI.instance.set_infoline(f'[b]t:[/b] {simt} [b]dt:[/b] {simdt} [b]Speed:[/b] {speed:.1f} [b]UTC:[/b] {simutc} [b]Mode:[/b] {self.modes[state]} [b]Aircraft:[/b] {ntraf}')

            # concate times of all nodes
            node_times = "".join([self.nodes[key]['time'] for key in self.nodes])

            # send to tui
            ConsoleUI.instance.set_nodes(copy.deepcopy(self.nodes), node_times)
        
        if name == b'ACDATA':
            self.extend_node_data(data, sender_id)
            if sender_id == bs.net.actnode():
                gen_data, table_data = self.get_traffic(data)
                ConsoleUI.instance.set_traffic(gen_data, table_data, sender_id)

    def extend_node_data(self, data, connid):
        # check if it is inside self.nodes
        if connid in self.nodes:
            self.nodes[connid]['nair'] = len(data['id'])
            self.nodes[connid]['nconf_cur'] = data['nconf_cur']
            self.nodes[connid]['nconf_tot'] = data['nconf_tot']
            self.nodes[connid]['nlos_cur'] = data['nlos_cur']
            self.nodes[connid]['nlos_tot'] = data['nlos_tot']

    def get_traffic(self, data):
        
        # general data
        gen_data = dict()
        gen_data['simt'] = data['simt'] 
        
        # only keep some info for table
        table_data = dict()

        table_data['id'] = data['id']
        table_data['lat'] = np.array(np.round(data['lat'],4), dtype=str)
        table_data['lon'] = np.array(np.round(data['lon'],4), dtype=str)
        table_data['alt (ft)'] = np.array(np.round(data['alt']/ft, 0), dtype=str)
        table_data['tas (kts)'] = np.array(np.round(data['tas']/kts,2), dtype=str)
        table_data['cas (kts)'] = np.array(np.round(data['cas']/kts,2), dtype=str)
        table_data['inconf'] = np.array(np.array(data['inconf'], dtype=bool), dtype=str)
        table_data['tcpamax (s)'] = np.array(np.round(data['tcpamax'],2), dtype=str)
        table_data['rpz (nm)'] = np.array(np.round(data['rpz']/nm,2), dtype=str)
        table_data['vs (fpm)'] = np.array(np.round(data['vs']/fpm,2), dtype=str)
        table_data['vmin (kts)'] = np.array(np.round(data['vmin']/kts,2), dtype=str)
        table_data['vmax (kts)'] = np.array(np.round(data['vmax']/kts,2), dtype=str)

        return gen_data, table_data
        
    def setNodeInfo(self, connid, time, scenname):
        
        # check if it is inside self.nodes
        if connid in self.nodes:
            self.nodes[connid]['scenename'] = scenname
            self.nodes[connid]['time'] = time
        
        else:
            node_count = self.count + 1
            self.nodes[connid] = {'num': f'{node_count}','scenename': scenname, 
                                'time': time, 'nair': 0, 'nconf_cur': 0, 'nconf_tot': 0,
                                'nlos_cur': 0, 'nlos_tot': 0}
            self.count += 1
            
class Echobox(ScrollView):
    text = Reactive("")

    async def watch_text(self, text):
        await self.update(Panel(Text(text), height=max(8, 2 + text.count('\n')), box=box.SIMPLE, style=Style(bgcolor="grey53")))

    def set_text(self, text: str):
        self.text = text

class Textline(Widget):
    text = Reactive("")
    style = Style(bgcolor="grey37")

    def __init__(self, text="", name=None):
        super().__init__(name)
        self.text = text

    def render(self) -> RenderableType:
        return self.text

    def set_text(self, text):
        self.text = text

class NodeInfo(Widget):
    table: Union[Reactive[Table], Table] = Reactive(Table())

    def __init__(self, name=None):
        super().__init__(name)
        self.table = Table()

        self.regular_style = Style(color='bright_green', bgcolor="bright_black")
        self.actnode_style = Style(color='bright_yellow', bgcolor="black")

    def render(self) -> RenderableType:
        return self.table

    def set_nodes(self, nodedict: dict):
        # add rows to table
        self.table = Table(title='[bold blue]Nodes', box=box.MINIMAL_DOUBLE_HEAD)
        self.table.add_column('Node #', header_style=Style(color="magenta"))
        self.table.add_column('Scenario', header_style=Style(color="magenta"))
        self.table.add_column('Time', header_style=Style(color="magenta"))
        self.table.add_column('Aircraft', header_style=Style(color="magenta"))
        self.table.add_column('Current conflicts', header_style=Style(color="magenta"))
        self.table.add_column('Total conflicts', header_style=Style(color="magenta"))
        self.table.add_column('Current LOS', header_style=Style(color="magenta"))
        self.table.add_column('Total LOS', header_style=Style(color="magenta"))

        for node_id, node in nodedict.items():
            
            if bs.net.act == node_id:
               self.table.add_row(node['num'], node['scenename'], node['time'],
                                    f'{node["nair"]}',
                                    f'{node["nconf_cur"]}', f'{node["nconf_tot"]}', 
                                    f'{node["nlos_cur"]}', f'{node["nlos_tot"]}', 
                                style=self.actnode_style)
            else:
                self.table.add_row(node['num'], node['scenename'], node['time'],
                                    f'{node["nair"]}',
                                    f'{node["nconf_cur"]}', f'{node["nconf_tot"]}', 
                                    f'{node["nlos_cur"]}', f'{node["nlos_tot"]}',
                                style=self.regular_style)

class Traffic(Widget):
    table: Union[Reactive[Table], Table] = Reactive(Table())

    def __init__(self, name=None):
        super().__init__(name)
        self.table = Table()

    def render(self) -> RenderableType:
        return self.table

    def set_traffic(self, trafict: dict, active_node_num: float):
        # add rows to table
        self.table = Table(title=f'[bold blue]Traffic for node # {active_node_num}', box=box.MINIMAL_DOUBLE_HEAD)

        # add the columns
        for col in trafict.keys():
            self.table.add_column(col, header_style=Style(color="magenta"))
        
        ntraf = len(trafict['id'])

        if ntraf > 0:
            for i in range(ntraf):
                # limit traffic to 100 aircraft on screen
                if i > 100:
                    break
                row = []
                for col in trafict.keys():
                    table_value = trafict[col][i]
                    row.append(table_value)
                self.table.add_row(*row, style=Style(color='bright_green' ,bgcolor="bright_black"))

class NodeTree(TreeControl):
    nodenums = Reactive("")
    nnodes = Reactive(0)

    def __init__(self, label=None, *args):
        super().__init__(label, *args)
        self.nnodes = 0
        self.original_label = label
        self.root.tree.guide_style = "green4"
        self._tree.style = Style(color="deep_sky_blue1")

    def __reinit__(self):
        # reinitialize tree for new nodes
        self._tree.label = self.root
        self.data = {}
        self.id = NodeID(0)
        self.nodes: dict[NodeID, TreeNode[NodeDataType]] = {}
        self._tree = Tree(self.original_label)
        self.root: TreeNode[NodeDataType] = TreeNode(
            None, self.id, self, self._tree, self.original_label, {}
        )
        self._tree.label = self.root
        self.nodes[NodeID(self.id)] = self.root
        self.root.tree.guide_style = "blue"
        self._tree.style = Style(color="deep_sky_blue1")

    async def handle_tree_click(self, message: TreeClick[dict]):
        """Called in response to a tree click."""
        new_active_node = message.node.data.get("node_id")
        # convert active node string to bytes
        bs.net.actnode(new_active_node)

    async def watch_nnodes(self, nnodes):

        # reinitialize if nodes exist
        if list(self.nodes.keys())[1:]:
            self.__reinit__()

        # reload nodes
        for node in self.nodenums:
            node_id = self.node_ids[self.nodenums.index(node)]
            await self.add(self.root.id, node, {"node_num": node, "node_id": node_id})
        await self.root.expand()

    def set_nodedict(self, nodedict):

        self.nodenums = []
        self.node_ids = []
        for node_id, node in nodedict.items():
            nodenum, node_id = node['num'], node_id
            self.nodenums.append(f'Node {nodenum}')
            self.node_ids.append(node_id)
        self.nnodes = len(nodedict)


class ConsoleUI(App):
    cmdtext: Union[Reactive[str], str] = Reactive("")
    echotext: Union[Reactive[str], str] = Reactive("")
    infotext: Union[Reactive[str], str] = Reactive("")
    nodedict = Reactive(dict())
    nodetimes = Reactive("0")
    trafsimt = Reactive("")
    ntraf = Reactive(0)
    nnodes = Reactive(0)
    active_node_num = Reactive(0)

    cmdbox: Textline
    echobox: Echobox
    infoline: Textline
    nodeinfo: NodeInfo
    traffic: Traffic
    tree: NodeTree
    instance: App
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ConsoleUI.instance = self
        
    def echo(self, text, flags=None):
        self.echotext = text + '\n' + self.echotext

    def set_infoline(self, text):
        self.infotext = text
        
    def set_nodes(self, nodes, nodetimes):
        self.nodedict = nodes
        self.nnodes = len(nodes)
        self.nodetimes = nodetimes

    def set_traffic(self, gen_data, traffic, active_node):
        self.trafdict = traffic
        self.trafsimt = str(gen_data['simt'])
        self.ntraf = len(traffic['id'])
        self.active_node_num = self.nodedict[active_node]['num'] if self.nodedict else 1
        
    async def on_key(self, key: events.Key):
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

    async def watch_infotext(self, infotext):
        self.infoline.set_text(f"[black]Current node:[/black] {infotext}")

    async def watch_cmdtext(self, cmdtext):
        self.cmdbox.set_text(f"[blue]>>[/blue] {cmdtext}")
        
    async def watch_nodetimes(self, nodetimes):
        self.nodeinfo.set_nodes(self.nodedict)
        self.tree.set_nodedict(self.nodedict)
    
    async def watch_nnodes(self, nnodes):
        await self.nodebody.update(self.nodeinfo)

    async def watch_echotext(self, echotext):
        self.echobox.set_text(echotext)

    async def watch_ntraf(self, ntraf):
        await self.trafbody.update(self.traffic)

    async def watch_trafsimt(self, trafsimt):
        self.traffic.set_traffic(self.trafdict, self.active_node_num)
    
    async def watch_active_node_num(self, active_node_num):
        self.nodeinfo.set_nodes(self.nodedict)

    async def on_mount(self, event: events.Mount):
        self.cmdbox = Textline("[blue]>>[/blue]")
        self.echobox = Echobox(Panel(Text(), height=8, box=box.SIMPLE, style=Style(bgcolor="grey53")))
        self.infoline = Textline("[black]Current node: [/black]")
        self.nodeinfo = NodeInfo(name="nodeinfo")
        self.traffic = Traffic(name="traffic")
        self.tree = NodeTree("Switch Nodes", {})

        await self.bind(Keys.Escape, "quit", "Quit")
        await self.bind(Keys.ControlT, "view.toggle('trafficbody')", "Show traffic")
        await self.bind(Keys.ControlB, "view.toggle('nodedock')", "Show batch")

        await self.view.dock(Footer(), edge="bottom", size=1)
        await self.view.dock(self.cmdbox, edge="bottom", size=1)

        echorow = DockView()
        self.treebody = ScrollView(self.tree)
        await echorow.dock(self.treebody, edge="right", size=20)
        await echorow.dock(self.echobox, edge="left")
        
        await self.view.dock(echorow, edge="bottom", size=8)
        await self.view.dock(self.infoline, edge="bottom", size=1)
        
        self.trafbody = ScrollView(self.traffic, name="trafficbody")
        await self.view.dock(self.trafbody, edge='top')

        self.nodebody = ScrollView(self.nodeinfo, name="nodeinfobody")
        await self.view.dock(self.nodebody, edge='top')

        await self.set_focus(self.cmdbox)

        self.set_interval(0.2, bs.net.update, name='Network')

def start(hostname=None):
    ''' Create and start BlueSky text-based client. '''
    bsclient = ConsoleClient()
    bsclient.connect(hostname=hostname)

    ConsoleUI.run(log="textual.log")