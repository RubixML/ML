<?php

namespace Rubix\Engine;

use InvalidArgumentException;
use RuntimeException;
use SplObjectStorage;
use SplPriorityQueue;
use SplQueue;
use SplStack;

class Graph
{
    /**
     * An index of all the nodes in the graph.
     *
     * @var  \Rubix\Engine\ObjectIndex
     */
    protected $nodes;

    /**
     * Factory method to build a graph from a list of nodes and edges.
     *
     * @param  array  $nodes
     * @param  array  $edges
     * @return self
     */
    public static function build(array $nodes = [], array $edges = []) : Graph
    {
        $graph = new self();

        foreach ($nodes as $node) {
            $graph->insert($node['id'], $node['properties'] ?? []);
        }

        foreach ($edges as $edge) {
            $graph->find($edge['start'])->attach($graph->find($edge['end']), $edge['properties'] ?? []);
        }

        return $graph;
    }

    /**
     * @param  \Rubix\Engine\ObjectIndex|null  $nodes
     * @return void
     */
    public function __construct(ObjectIndex $nodes = null)
    {
        if (is_null($nodes)) {
            $nodes = new ObjectIndex();
        }

        $this->nodes = $nodes;
    }

    /**
     * @return \Rubix\Engine\ObjectIndex
     */
    public function nodes() : ObjectIndex
    {
        return $this->nodes;
    }

    /**
     * Insert a node into the graph. O(1)
     *
     * @param  mixed|null  $id
     * @param  array  $properties
     * @param  bool  $overwrite
     * @throws \RuntimeException
     * @return \Rubix\Engine\Node
     */
    public function insert($id, array $properties = [], bool $overwrite = false) : Node
    {
        if ($overwrite === false) {
            if ($this->nodes->has($id)) {
                throw new RuntimeException('Node with ID ' . (string) $id . ' already exists in the graph.');
            }
        }

        $node = new Node($id, $properties);

        $this->nodes->put($id, $node);

        return $node;
    }

    /**
     * Find a node in the graph.
     *
     * @param  mixed  $id
     * @return \Rubix\Engine\Node
     */
    public function find($id) : ?Node
    {
        return $this->nodes->get($id);
    }

    /**
     * Find many nodes.
     *
     * @param  array  $ids
     * @return array
     */
    public function findMany(array $ids) : array
    {
        return $this->nodes->mget($ids);
    }

    /**
     * Find a path between a start node to an end node.
     * Returns null if no path can be found. O(V+E)
     *
     * @param  \Rubix\Engine\Node  $start
     * @param  \Rubix\Engine\Node  $end
     * @return \Rubix\Engine\Path|null
     */
    public function findPath(Node $start, Node $end) : ?Path
    {
        $discovered = new SplObjectStorage();
        $stack = new SplStack();
        $path = new Path();

        $discovered->attach($start, null);

        $stack->push($start);

        while (!$stack->isEmpty()) {
            $current = $stack->pop();

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

                    $stack->push($edge->node());
                }
            }
        }

        return null;
    }

    /**
     * Find a shortest path between a start node to an end node.
     * Returns null if no path can be found. O(V+E)
     *
     * @param  \Rubix\Engine\Node  $start
     * @param  \Rubix\Engine\Node  $end
     * @return \Rubix\Engine\Path|null
     */
    public function findShortestPath(Node $start, Node $end) : ?Path
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
     * Find a shortest weighted path between start node and an end node.
     * Returns null if no path can be found. O(V*E)
     *
     * @param  \Rubix\Engine\Node  $start
     * @param  \Rubix\Engine\Node  $end
     * @param  string  $weight
     * @param  mixed  $default
     * @throws \RuntimeException
     * @return \Rubix\Engine\Path|null
     */
    public function findShortestWeightedPath(Node $start, Node $end, string $weight, $default = INF) : ?Path
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

        foreach (range(1, $this->nodes->count()) as $iteration) {
            foreach ($this->nodes() as $current) {
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
     * Returns null if no path can be found. O(V+ElogV)
     *
     * @param  \Rubix\Engine\Node  $start
     * @param  \Rubix\Engine\Node  $end
     * @param  string  $relationship
     * @param  string  $weight
     * @param  mixed  $default
     * @return \Rubix\Engine\Path|null
     */
    public function findShortestUnsignedWeightedPath(Node $start, Node $end, string $weight, $default = INF) : ?Path
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

                    $queue->insert($edge->node(), 0 - $distance);
                }
            }
        }

        return null;
    }

    /**
     * Topologically sort the graph. O(V+E)
     *
     * @param  string  $relationship
     * @return \Rubix\Engine\Path
     */
    public function sort() : Path
    {
        $discovered = new SplObjectStorage();
        $stack = new SplStack();
        $path = new Path();

        foreach ($this->nodes as $node) {
            $stack->push($node);
        }

        while (!$stack->isEmpty()) {
            $current = $stack->pop();

            if (!$discovered->contains($current)) {
                $discovered->attach($current);

                $stack->push($current);

                foreach ($current->edges() as $edge) {
                    $stack->push($edge->node());
                }
            } else {
                $path->prepend($current);
            }
        }

        return $path;
    }

    /**
     * Is the relationship acyclic? O(V+E)
     *
     * @return bool
     */
    public function acyclic() : bool
    {
        return !$this->cyclic();
    }

    /**
     * Does the graph contain an infinite cycle. O(V+E)
     *
     * @return bool
     */
    public function cyclic() : bool
    {
        foreach ($this->nodes as $node) {
            $discovered = new SplObjectStorage();
            $stack = new SplStack();

            $stack->push($node);

            while (!$stack->isEmpty()) {
                $current = $stack->pop();

                foreach ($current->edges() as $edge) {
                    if ($discovered->contains($edge->node())) {
                        return true;
                    }

                    $discovered->attach($edge->node());
                    $stack->push($edge->node());
                }
            }
        }

        return false;
    }

    /**
     * Remove a node from the graph. O(N)
     *
     * @param  mixed  $id
     * @return self
     */
    public function delete($id) : Graph
    {
        foreach ($this->nodes as $current) {
            $current->edges()->remove($id);
        }

        $this->nodes->remove($id);

        return $this;
    }
}
