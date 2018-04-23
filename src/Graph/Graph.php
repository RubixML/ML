<?php

namespace Rubix\Engine\Graph;

use Rubix\Engine\Persisters\Persister;
use Rubix\Engine\Persisters\Persistable;
use RuntimeException;
use SplObjectStorage;
use SplPriorityQueue;
use Countable;
use SplQueue;

class Graph implements Persistable, Countable
{
    /**
     * An index of the nodes in the graph.
     *
     * @var \Rubix\Engine\ObjectIndex
     */
    protected $nodes;

    /**
     * Restore the graph from persistence.
     *
     * @param \Rubix\Engine\Persisters\Persister  $connector
     * @return self|null
     */
    public static function restore(Connector $connector) : self
    {
        $graph = $connector->restore();

        if (!$graph instanceof Graph) {
            throw new InvalidargumentException('The restored model is not a Graph instance.');
        }

        return $graph;
    }

    /**
     * @return void
     */
    public function __construct(Connector $connector = null)
    {
        $this->nodes = new ObjectIndex();
        $this->connector = $connector;
    }

    /**
     * The order of the graph, or the total number of nodes. O(1)
     *
     * @return int
     */
    public function order() : int
    {
        return $this->count();
    }

    /**
     * The size of the graph, or the total number of edges. O(V)
     *
     * @return int
     */
    public function size() : int
    {
        return array_reduce($this->nodes->all(), function ($carry, $node) {
            return $carry += $node->edges()->count();
        }, 0);
    }

    /**
     * @return \Rubix\Engine\ObjectIndex
     */
    public function nodes() : ObjectIndex
    {
        return $this->nodes;
    }

    /**
     * Insert a node into the layer. O(1)
     *
     * @param  array  $properties
     * @return \Rubix\Engine\GraphNode
     */
    public function insert(array $properties = []) : GraphNode
    {
        $node = new GraphNode($properties);

        $this->nodes->put($node->id(), $node);

        return $node;
    }

    /**
     * Find a node in the graph by ID. O(1)
     *
     * @param  int  $id
     * @return \Rubix\Engine\GraphNode
     */
    public function find(int $id) : ?GraphNode
    {
        return $this->nodes->get($id);
    }

    /**
     * Find many nodes in the graph by ID. Returns an array indexed by the node ID.
     *
     * @param  array  $ids
     * @return array
     */
    public function findMany(array $ids) : array
    {
        return $this->nodes->mget($ids);
    }

    /**
     * Find a path between a start node to an end node. Returns null if no path can
     * be found. O(V+E)
     *
     * @param  \Rubix\Engine\GraphNode  $start
     * @param  \Rubix\Engine\GraphNode  $end
     * @return \Rubix\Engine\Path|null
     */
    public function findPath(GraphNode $start, GraphNode $end) : ?Path
    {
        $discovered = new SplObjectStorage();
        $path = new Path();
        $stack = [];

        $discovered->attach($start, null);

        array_push($stack, $start);

        while (!empty($stack)) {
            $current = array_pop($stack);

            if ($current->isSame($end)) {
                while ($end !== null) {
                    $path->prepend($end);

                    $end = $discovered[$end];
                }

                return $path;
            }

            foreach ($current->edges() as $edge) {
                $node = $edge->node();

                if (!$discovered->contains($node)) {
                    $discovered->attach($node, $current);

                    $stack[] = $node;
                }
            }
        }

        return null;
    }

    /**
     * Find all paths between a start node to an end node. Returns an empty array
     * if no paths are found.
     *
     * @param  \Rubix\Engine\GraphNode  $start
     * @param  \Rubix\Engine\GraphNode  $end
     * @return array
     */
    public function findAllPaths(GraphNode $start, GraphNode $end) : array
    {
        $paths = [];

        $this->_findAllPaths($start, $end, new SplObjectStorage(), new Path(), $paths);

        return $paths;
    }

    /**
     * Recursive backtracking function to find all paths between two given nodes.
     *
     * @param  \Rubix\Engine\GraphNode  $root
     * @param  \Rubix\Engine\GraphNode  $end
     * @param  \SplObjectStorage  $discovered
     * @param  \Rubix\Engine\Path  $path
     * @param  array  $paths
     * @return void
     */
    protected function _findAllPaths(GraphNode $root, GraphNode $end, SplObjectStorage $discovered, Path $path, array &$paths) : void
    {
        $discovered->attach($root);
        $path->push($root);

        if ($root->isSame($end)) {
            $paths[] = clone $path;
        } else {
            foreach ($root->edges() as $edge) {
                $node = $edge->node();

                if (!$discovered->contains($node)) {
                    $this->_findAllPaths($node, $end, $discovered, $path, $paths);
                }
            }
        }

        $discovered->detach($root);
        $path->pop();
    }

    /**
     * Find a shortest path between a start node and an end node. Returns null if
     * no path can be found. O(V+E)
     *
     * @param  \Rubix\Engine\GraphNode  $start
     * @param  \Rubix\Engine\GraphNode  $end
     * @return \Rubix\Engine\Path|null
     */
    public function findShortestPath(GraphNode $start, GraphNode $end) : ?Path
    {
        $discovered = new SplObjectStorage();
        $queue = new SplQueue();
        $path = new Path();

        $discovered->attach($start, null);

        $queue->enqueue($start);

        while (!$queue->isEmpty()) {
            $current = $queue->dequeue();

            if ($current->isSame($end)) {
                while ($end !== null) {
                    $path->prepend($end);

                    $end = $discovered[$end];
                }

                return $path;
            }

            foreach ($current->edges() as $edge) {
                if (!$discovered->contains($edge->node())) {
                    $discovered->attach($edge->node(), $current);

                    $queue->enqueue($edge->node());
                }
            }
        }

        return null;
    }

    /**
     * Find a shortest path between each node in the graph. Returns an array of
     * paths in order they were discovered. O(V^2+E)
     *
     * @return array
     */
    public function findAllPairsShortestPaths() : array
    {
        $paths = [];

        foreach ($this->nodes as $start) {
            foreach ($this->nodes as $end) {
                if (!$start->isSame($end)) {
                    $path = $this->findShortestPath($start, $end);

                    if (isset($path)) {
                        $paths[] = $path;
                    }
                }
            }
        }

        return $paths;
    }

    /**
     * Find a shortest weighted path between start node and an end node.
     * Returns null if no path can be found. O(V*E)
     *
     * @param  \Rubix\Engine\GraphNode  $start
     * @param  \Rubix\Engine\GraphNode  $end
     * @param  string  $weight
     * @param  mixed  $default
     * @throws \RuntimeException
     * @return \Rubix\Engine\Path|null
     */
    public function findShortestWeightedPath(GraphNode $start, GraphNode $end, string $weight, $default = INF) : ?Path
    {
        $discovered = new SplObjectStorage();
        $path = new Path();

        foreach ($this->nodes as $node) {
            $discovered->attach($node, [
                'parent' => null,
                'distance' => INF,
            ]);
        }

        $discovered[$start] = [
            'parent' => null,
            'distance' => 0,
        ];

        foreach (range(1, $this->nodes->count() - 1) as $i) {
            foreach ($this->nodes as $current) {
                foreach ($current->edges() as $edge) {
                    $distance = $discovered[$current]['distance'] + $edge->get($weight, $default);

                    if ($discovered[$current]['distance'] != INF && $distance < $discovered[$edge->node()]['distance']) {
                        $discovered[$edge->node()] = [
                            'parent' => $current,
                            'distance' => $distance,
                        ];
                    }
                }
            }
        }

        // Run the algorithm one more time, if we are still able to relax an edge,
        // then it is an infinite negative weight cycle and no shortest path can
        // be computed.
        foreach ($this->nodes as $current) {
            foreach ($current->edges() as $edge) {
                if ($discovered[$current]['distance'] != INF && $discovered[$current]['distance'] + $edge->$weight < $discovered[$edge->node()]['distance']) {
                    throw new RuntimeException('Graph contains an infinite negative weight cycle.');
                }
            }
        }

        while ($end !== null) {
            $path->prepend($end);

            $end = $discovered[$end]['parent'];
        }

        if ($path->first()->isSame($start)) {
            return $path;
        }

        return null;
    }

    /**
     * Find a shortest unsigned weighted path between start node and an end node.
     * Returns null if no path can be found. O(VlogV+ElogV)
     *
     * @param  \Rubix\Engine\GraphNode  $start
     * @param  \Rubix\Engine\GraphNode  $end
     * @param  string  $weight
     * @param  mixed  $default
     * @return \Rubix\Engine\Path|null
     */
    public function findShortestUnsignedWeightedPath(GraphNode $start, GraphNode $end, string $weight, $default = INF) : ?Path
    {
        $discovered = new SplObjectStorage();
        $queue = new SplPriorityQueue();
        $path = new Path();

        foreach ($this->nodes as $node) {
            $discovered->attach($node, [
                'parent' => null,
                'distance' => INF,
            ]);
        }

        $discovered[$start] = [
            'parent' => null,
            'distance' => 0,
        ];

        $queue->insert($start, 0);

        while (!$queue->isEmpty()) {
            $current = $queue->extract();

            if ($current->isSame($end)) {
                while ($end !== null) {
                    $path->prepend($end);

                    $end = $discovered[$end]['parent'];
                }

                return $path;
            }

            foreach ($current->edges() as $edge) {
                $distance = $discovered[$current]['distance'] + abs($edge->get($weight, $default));

                if ($distance < $discovered[$edge->node()]['distance']) {
                    $discovered[$edge->node()] = [
                        'parent' => $current,
                        'distance' => $distance,
                    ];

                    $queue->insert($edge->node(), -$distance);
                }
            }
        }

        return null;
    }

    /**
     * Find a shortest unsigned weighted path between each node in the graph. Returns
     * an array of paths in order they were discovered.
     *
     * @param  string  $weight
     * @param  string  $default
     * @return array
     */
    public function findAllPairsShortestUnsignedWeightedPath(string $weight, $default = INF) : array
    {
        $paths = [];

        foreach ($this->nodes as $start) {
            foreach ($this->nodes as $end) {
                if (!$start->isSame($end)) {
                    $path = $this->findShortestUnsignedWeightedPath($start, $end, $weight, $default);

                    if (isset($path)) {
                        $paths[] = $path;
                    }
                }
            }
        }

        return $paths;
    }

    /**
     * Return a path of topologically sorted nodes which will only be valid if
     * the graph is acyclic. Returns null if graph is empty. O(V+E)
     *
     * @return \Rubix\Engine\Path|null
     */
    public function sort() : ?Path
    {
        if ($this->order() === 0) {
            return null;
        }

        $discovered = new SplObjectStorage();
        $path = new Path();
        $stack = [];

        foreach ($this->nodes as $node) {
            $stack[] = $node;
        }

        while (!empty($stack)) {
            $current = array_pop($stack);

            if (!$discovered->contains($current)) {
                $discovered->attach($current);

                $stack[] = $current;

                foreach ($current->edges() as $edge) {
                    $stack[] = $edge->node();
                }
            } else {
                $path->prepend($current);
            }
        }

        return $path;
    }

    /**
     * Is the graph acyclic? O(V+E)
     *
     * @return bool
     */
    public function acyclic() : bool
    {
        return !$this->cyclic();
    }

    /**
     * Does the graph contain at least one infinite cycle. O(V+E)
     *
     * @return bool
     */
    public function cyclic() : bool
    {
        foreach ($this->nodes as $node) {
            $discovered = new SplObjectStorage();
            $stack = [];

            array_push($stack, $node);

            while (!empty($stack)) {
                $current = array_pop($stack);

                foreach ($current->edges() as $edge) {
                    if ($discovered->contains($edge->node())) {
                        return true;
                    }

                    $discovered->attach($edge->node());

                    $stack[] = $edge->node();
                }
            }
        }

        return false;
    }

    /**
     * Save the graph to persistence.
     *
     * @return bool
     */
    public function save() : bool
    {
        return $this->connector->save($this);
    }

    /**
     * Remove a node from the graph by ID.
     *
     * @param  int  $id
     * @return self
     */
    public function delete(int $id) : self
    {
        $node = $this->find($id);

        if (isset($node)) {
            foreach ($node->edges() as $edge) {
                $edge->node()->detach($node);
            }

            $this->nodes->remove($id);
        }

        return $this;
    }

    /**
     * The number of nodes in the layer.
     *
     * @return int
     */
    public function count() : int
    {
        return $this->nodes->count();
    }
}
