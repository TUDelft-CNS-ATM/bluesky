#!/usr/bin/python

"""
We create a class to define a MultiDiGrpah 3D. We add several useful methods
"""

__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'

import string

from networkx.classes.multidigraph import MultiDiGraph

import networkx as nx
import osmnx as ox


class MultiDiGrpah3D ( MultiDiGraph ):
    def __init__( self, G_copy=None, **attr ):
        MultiDiGraph.__init__( self, G_copy, **attr )

    def addNodeAltitude( self, name, y_node, x_node, alttude_node ):
        self.add_node( name, y=y_node, x=x_node, z=alttude_node, segment='new' )

    def filterNodes( self, nodes, layer ):
        nodes_filtered = filter( lambda node: str( node )[0] == layer, nodes )
        return list( nodes_filtered )

    def defGroundAltitude( self ):
        for n in self:
            self.add_node( n, z=0 )

    def addLayer( self, layer, layer_width ):
        increment_altitude = layer_width

        nodes = list( self.nodes )

        if layer == 'A':  # If we are creating the first layer (A)
            below_layer = ''
            # We select the layer below
            nodes_filtered = nodes

            # We add nodes of the new layer
            for n in nodes_filtered:
                name = layer + str( n )

                self.addNodeAltitude( name, self._node[n]["y"], self._node[n]["x"],
                                      self._node[n]["z"] + increment_altitude )

            # We connect the nodes of the new layer
            for n in nodes_filtered:
                name = layer + str( n )
                connected_edges = list( self.neighbors( n ) )

                for j in connected_edges:
                    dest = "A" + str( j )

                    self.add_edge( name, dest, 0, oneway=self.edges[n, j, 0]['oneway'],
                                   segment='new', speed=50.0,
                                   length=self.edges[n, j, 0]['length'] )
                    self.add_edge( dest, name, 0, oneway=self.edges[n, j, 0]['oneway'],
                                   segment='new', speed=50.0,
                                   length=self.edges[n, j, 0]['length'] )

            # The new layer is connected with the layer below
            for n in nodes_filtered:
                name = "A" + str( n )
                self.add_edge( n, name, 0, oneway=False, segment='new', speed=50.0,
                               length=increment_altitude )
                self.add_edge( name, n, 0, oneway=False, segment='new', speed=50.0,
                               length=increment_altitude )

        else:  # if we are adding a layer which is not the first one
            # We select the layer below
            below_layer = chr( ord( layer ) - 1 )
            nodes_filtered = self.filterNodes( nodes, below_layer )

            # We add nodes of the new layer
            for n in nodes_filtered:
                name = layer + str( n )[1:]

                self.addNodeAltitude( name, self._node[n]["y"], self._node[n]["x"],
                                      self._node[n]["z"] + increment_altitude )

            # We connect the nodes of the new layer
            for n in nodes_filtered:
                name = layer + str( n )[1:]
                connected_edges = list( self.neighbors( n ) )
                connected_edges_filtered = self.filterNodes( connected_edges, below_layer )

                for j in connected_edges_filtered:
                    dest = layer + str( j )[1:]

                    self.add_edge( name, dest, 0, oneway=self.edges[n, j, 0]['oneway'],
                                   segment='new', speed=50.0,
                                   length=self.edges[n, j, 0]['length'] )

            # The new layer is connected with the layer below
            for n in nodes_filtered:
                name = layer + str( n )[1:]
                self.add_edge( n, name, 0, oneway=False, segment='new',
                               speed=50.0, length=increment_altitude )
                self.add_edge( name, n, 0, oneway=False, segment='new',
                               speed=50.0, length=increment_altitude )

    def addDiagonalEdges( self, sectors, config ):
        print( 'Creating diagonal edges...' )
        for i in range( len( sectors ) ):
            # The four corners of each sector
            point1_x = sectors.loc[[i], ['lon_min']].values[0][0]
            point1_y = sectors.loc[[i], ['lat_min']].values[0][0]
            point2_x = sectors.loc[[i], ['lon_max']].values[0][0]
            point2_y = sectors.loc[[i], ['lat_min']].values[0][0]
            point3_x = sectors.loc[[i], ['lon_min']].values[0][0]
            point3_y = sectors.loc[[i], ['lat_max']].values[0][0]
            point4_x = sectors.loc[[i], ['lon_max']].values[0][0]
            point4_y = sectors.loc[[i], ['lat_max']].values[0][0]

            # The nodes that will be joined
            node1 = ox.distance.nearest_nodes( self, X=point1_x, Y=point1_y )
            node2 = ox.distance.nearest_nodes( self, X=point2_x, Y=point2_y )
            node3 = ox.distance.nearest_nodes( self, X=point3_x, Y=point3_y )
            node4 = ox.distance.nearest_nodes( self, X=point4_x, Y=point4_y )

            # Nodes in the middle of the edges
#             node12 = ox.distance.nearest_nodes( self, X=( point1_x + point2_x ) / 2,
#                                                 Y=( point1_y + point2_y ) / 2 )
#             node23 = ox.distance.nearest_nodes( self, X=( point2_x + point3_x ) / 2,
#                                                 Y=( point2_y + point3_y ) / 2 )
#             node34 = ox.distance.nearest_nodes( self, X=( point3_x + point4_x ) / 2,
#                                                 Y=( point3_y + point4_y ) / 2 )
#             node41 = ox.distance.nearest_nodes( self, X=( point4_x + point1_x ) / 2,
#                                                 Y=( point4_y + point1_y ) / 2 )
#             node1234 = ox.distance.nearest_nodes( self, X=( point1_x + point2_x + point3_x +
#                                                             point4_x ) / 4,
#                                                             Y=( point1_y + point2_y +
#                                                                 point3_y + point4_y ) / 4 )

            y1 = self.nodes[node1]['y']
            x1 = self.nodes[node1]['x']
            y2 = self.nodes[node2]['y']
            x2 = self.nodes[node2]['x']
            y3 = self.nodes[node3]['y']
            x3 = self.nodes[node3]['x']
            y4 = self.nodes[node4]['y']
            x4 = self.nodes[node4]['x']

#             y12 = self.nodes[node12]['y']
#             x12 = self.nodes[node12]['x']
#             y23 = self.nodes[node23]['y']
#             x23 = self.nodes[node23]['x']
#             y34 = self.nodes[node34]['y']
#             x34 = self.nodes[node34]['x']
#             y41 = self.nodes[node41]['y']
#             x41 = self.nodes[node41]['x']
#             y1234 = self.nodes[node1234]['y']
#             x1234 = self.nodes[node1234]['x']

            # Compute in which layers we have to add the edges
            altitude = config['Layers'].getint( 'number_of_layers' ) * \
                config['Layers'].getint( 'layer_width' )  # Altitude of the highest layer
            letters = list( string.ascii_uppercase )
            total_layers = letters[0:config['Layers'].getint( 'number_of_layers' )]
            layers = []
            while altitude > sectors.loc[[i], ['altitude_limit']].values[0][0]:
                if any( layers ):
                    layers.append( chr( ord( layers[-1] ) - 1 ) )
                else:
                    layers.append( total_layers[-1] )

                altitude -= config['Layers'].getint( 'layer_width' )

#             xx_orig = ( x1, x12, x2, x23, x3, x34, x4, x41,
#                         x1, x2, x3, x4, x12, x23, x34, x41 )
#             xx_dest = ( x12, x2, x23, x3, x34, x4, x41, x1, x1234,
#                         x1234, x1234, x1234, x1234, x1234, x1234 )
#             yy_orig = ( y1, y12, y2, y23, y3, y34, y4, y41,
#                         y1, y2, y3, y4, y12, y23, y34, y41 )
#             yy_dest = ( y12, y2, y23, y3, y34, y4, y41, y1, y1234,
#                         y1234, y1234, y1234, y1234, y1234, y1234 )
#             node_orig = ( node1, node12, node2, node23, node3, node34, node4, node41, node1,
#                             node2, node3, node4, node12, node23, node34, node41 )
#             node_dest = ( node12, node2, node23, node3, node34, node4, node41, node1, node1234,
#                           node1234, node1234, node1234, node1234, node1234, node1234 )

            xx_orig = ( x1, x2, x3, x4, x1, x2 )
            xx_dest = ( x2, x3, x4, x1, x3, x4 )
            yy_orig = ( y1, y2, y3, y4, y1, y2 )
            yy_dest = ( y2, y3, y4, y1, y3, y4 )
            node_orig = ( node1, node2, node3, node4, node1, node2 )
            node_dest = ( node2, node3, node4, node1, node3, node4 )

            for elem in layers:
                for x_orig_iter, x_dest_iter, y_orig_iter, y_dest_iter, orig_iter, dest_iter in zip( xx_orig, xx_dest, yy_orig, yy_dest, node_orig, node_dest ):

                    self.add_edge( elem + orig_iter[1:], elem + dest_iter[1:], 0, oneway=False,
                                   segment='new', speed=50.0,
                                   length=ox.distance.great_circle_vec( y_orig_iter, x_orig_iter,
                                                                        y_dest_iter, x_dest_iter ) )
                    self.add_edge( elem + dest_iter[1:], elem + orig_iter[1:], 0, oneway=False,
                                   segment='new', speed=50.0,
                                   length=ox.distance.great_circle_vec( y_orig_iter, x_orig_iter,
                                                                        y_dest_iter, x_dest_iter ) )

    def defOneWay( self, config ):
        letters = list( string.ascii_uppercase )
        total_layers = letters[0:config['Layers'].getint( 'number_of_layers' )]
        layers_w2e = total_layers[0::2]
        layers_e2w = total_layers[1::2]
        for node in self.nodes:
            node_lon = self.nodes[node]['x']
            connected_edges = list( self.neighbors( node ) )
            connected_edges_layer = self.filterNodes( connected_edges, node[0] )

            if node[0] in layers_w2e:
                for elem in connected_edges_layer:
                    elem_lon = self.nodes[elem]['x']
                    if elem_lon < node_lon:
                        self.remove_edge( node, elem, 0 )
            elif node[0] in layers_e2w:
                for elem in connected_edges_layer:
                    elem_lon = self.nodes[elem]['x']
                    if elem_lon > node_lon:
                        self.remove_edge( node, elem, 0 )

    def simplifyGraph( self, config ):
        nodes = list( self.nodes )
        for node in nodes:
            if self.has_node( node ):
                G = MultiDiGrpah3D( self )
                node_lon = self.nodes[node]['x']
                node_lat = self.nodes[node]['y']
                G.remove_node( node )
                while self.distanceNearestNode( G, node_lon, node_lat ) < config['Options'].getint( 'simplification_distance' ):
                    nearest_node = ox.distance.nearest_nodes( G, X=node_lon, Y=node_lat )
                    connected_edges = list( G.neighbors( nearest_node ) )
                    G.remove_node( nearest_node )
                    self.remove_node( nearest_node )
                    for elem in connected_edges:
                        self.add_edge( node, elem, 0, oneway=False, segment='new', speed=50.0,
                                       length=ox.distance.great_circle_vec( node_lat, node_lon,
                                                                            self.nodes[elem]['y'],
                                                                            self.nodes[elem]['x'] ) )
                        self.add_edge( elem, node, 0, oneway=False, segment='new', speed=50.0,
                                       length=ox.distance.great_circle_vec( node_lat, node_lon,
                                                                            self.nodes[elem]['y'],
                                                                            self.nodes[elem]['x'] ) )
            else:
                pass

    def distanceNearestNode( self, G, node_lon, node_lat ):
        node = ox.distance.nearest_nodes( G, X=node_lon, Y=node_lat )
        return ox.distance.great_circle_vec( node_lat, node_lon, self.nodes[node]['y'],
                                             self.nodes[node]['x'] )
