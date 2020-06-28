import networkx as nx


class SocialNetworkGraph:
    start_day = None
    end_day = None

    def __init__(self):
        self._G = nx.DiGraph()
        self.nodes = self._G.nodes

    def __getitem__(self, i):
        return self._G[i]

    def __len__(self):
        return len(self._G)

    def add_edges(self, edges):
        for edge in edges:
            data = self._G.get_edge_data(*edge)
            self._G.add_edge(*edge, weight=int(0 if data is None else data['weight']) + 1)

    def add_attribute(self, name, data):
        nx.set_node_attributes(self._G, data, name)

    def get_nodes_attribute(self, attr):
        d = nx.get_node_attributes(self._G, attr)
        return d

    def successors(self, node):
        if node not in self._G.nodes:
            return []
        return self._G.successors(node)

    def predecessors(self, node):
        if node not in self._G.nodes:
            return []
        return self._G.predecessors(node)

    def has_node(self, node):
        return self._G.has_node(node)

    def out_edges(self, node):
        return self._G.out_edges(node)

    def in_edges(self, node):
        return self._G.in_edges(node)

    def number_of_nodes(self):
        return self._G.number_of_nodes()

    def has_edge(self, u, v):
        return self._G.has_edge(u, v)

    def reciprocity(self, nodes=None):
        r = nx.algorithms.reciprocity(self._G, nodes if nodes else self.nodes)
        return {k: r[k] if r[k] is not None else 0 for k in r}

    def density(self, neighbors_function):
        return {n: nx.density(self._G.subgraph([n, *neighbors_function(n)])) for n in self._G.nodes}

    def degree(self, weight=False):
        if weight:
            return {node: sum([self._G[u][v]['weight'] for u, v in self._G.edges(node)]) for node in self.nodes}
        return {n: d for n, d in list(self._G.degree(self._G.nodes))}

    def in_degree(self, weight=False):
        if weight:
            return {node: sum([self._G[u][v]['weight'] for u, v in self._G.in_edges(node)]) for node in self.nodes}
        return {n: d for n, d in list(self._G.in_degree(self._G.nodes))}

    def out_degree(self, weight=False):
        if weight:
            return {node: sum([self._G[u][v]['weight'] for u, v in self._G.out_edges(node)]) for node in self.nodes}
        return {n: d for n, d in list(self._G.out_degree(self._G.nodes))}

    def in_degree_centrality(self):
        return nx.in_degree_centrality(self._G)

    def out_degree_centrality(self):
        return nx.out_degree_centrality(self._G)

    def degree_centrality(self):
        return nx.degree_centrality(self._G)

    def eigenvector_centrality(self, weight=False, reverse=False):
        if not weight:
            return nx.eigenvector_centrality(self._G if not reverse else self._G.reverse(), max_iter=1000)
        return nx.eigenvector_centrality(self._G if not reverse else self._G.reverse(), weight='weight', max_iter=1000)

    def katz_centrality(self, weight=False, reverse=False):
        if not weight:
            return nx.katz_centrality(self._G if not reverse else self._G.reverse(), max_iter=50000000)
        return nx.katz_centrality(self._G if not reverse else self._G.reverse(), weight='weight', max_iter=50000000)

    def closeness_centrality(self, reverse=False):
        return nx.closeness_centrality(self._G if not reverse else self._G.reverse())

    def betweenness_centrality(self):
        return nx.betweenness_centrality(self._G)

    def local_centrality(self, in_neighborhood=True):
        graph = self._G.reverse() if in_neighborhood else self._G
        N = {}  # Neighbors and nearest neighbors count
        for n in self._G.nodes:
            sp, _ = nx.single_source_dijkstra(graph, n, cutoff=2)
            del sp[n]
            N[n] = len(sp.keys())
        Q = {n: sum(N[s] for s in (self._G.predecessors(n) if in_neighborhood else self._G.successors(n)))
             for n in self._G.nodes}
        C = {n: sum(Q[s] for s in (self._G.predecessors(n) if in_neighborhood else self._G.successors(n)))
             for n in self._G.nodes}
        return C

    def jaccard_index_neighborhoods(self):
        return dict(self._jaccard_index_neighborhoods_iter())

    def _jaccard_index_neighborhoods_iter(self):
        """ Return an iterator of (node, jaccard_index).
        """
        for node in self._G.nodes:
            pred = set(self._G.predecessors(node))
            succ = set(self._G.successors(node))
            intersection = pred & succ

            if len(pred) == 0 or len(succ) == 0:
                yield (node, 0)
            else:
                jaccard_index = len(intersection)/(len(pred) + len(succ) - len(intersection))
                yield (node, jaccard_index)

    def neighborhood_fraction(self, neighbors_function):
        return dict(self._neighborhood_fraction_iter(neighbors_function))

    def _neighborhood_fraction_iter(self, neighbors_function):
        """ Return an iterator of (node, neighborhood_fraction).
        """
        for node in self._G.nodes:
            pred = set(self._G.predecessors(node))
            succ = set(self._G.successors(node))
            union = pred | succ

            diff = (pred, succ) if neighbors_function == self.predecessors else (succ, pred)
            difference = diff[0] - diff[1]

            if len(union) == 0:
                yield (node, 0)
            else:
                fraction = len(difference) / len(union)
                yield (node, fraction)

    def neighborhood_quality(self, neighbors_function, users_selection):
        return dict(self._neighborhood_quality_iter(neighbors_function, users_selection))

    def _neighborhood_quality_iter(self, neighbors_function, users_selection):
        """ Return an iterator of (node, neighborhood_quality).
        """
        for node in self._G.nodes:
            neigh = set(neighbors_function(node))
            intersection = neigh & set(users_selection)

            if len(intersection) == 0:
                yield (node, 0)
            else:
                quality = len(intersection) / len(users_selection)
                yield (node, quality)


