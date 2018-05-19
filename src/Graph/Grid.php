<?php

namespace Rubix\Engine\Graph;

use Rubix\Engine\Metrics\Distance\Euclidean;
use Rubix\Engine\Metrics\Distance\Distance;
use InvalidArgumentException;
use SplObjectStorage;
use SplPriorityQueue;

class Grid extends Graph
{
    /**
     * The axis labels of the grid.
     *
     * @var  array
     */
    protected $axes;

    /**
     * The distance function that describes the grid space.
     *
     * @var \Rubix\Engine\Metrics\Distance\Distance
     */
    protected $distanceFunction;

    /**
     * @param  array  $axes
     * @param  \Rubix\Engine\Contracts\Distance|null  $distanceFunction
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $axes = ['x', 'y'], Distance $distanceFunction = null)
    {
        if ($axes !== array_filter($axes, 'is_string')) {
            throw new InvalidArgumentException('Axis label must be a string type.');
        }

        if (!isset($distanceFunction)) {
            $distanceFunction = new Euclidean();
        }

        $this->axes = $axes;
        $this->distanceFunction = $distanceFunction;

        parent::__construct();
    }

    /**
     * Return the number of dimensions of the grid.
     *
     * @return int
     */
    public function dimensions() : int
    {
        return count($this->axes);
    }

    /**
     * The labels of the axes of the grid.
     *
     * @return array
     */
    public function labels() : array
    {
        return $this->axes;
    }

    /**
     * Insert a node into the grid. O(1)
     *
     * @param  array  $properties
     * @throws \InvalidArgumentException
     * @return \Rubix\Engine\GraphNode
     */
    public function insert(array $properties = []) : GraphNode
    {
        foreach ($this->axes as $axis) {
            if (!isset($properties[$axis])) {
                throw new InvalidArgumentException('Node must have a value set for all axis of the grid.');
            }
        }

        return parent::insert($properties);
    }

    /**
     * Compute the distance between two given nodes on the grid.
     *
     * @param  \Rubix\Engine\GraphNode  $start
     * @param  \Rubix\Engine\GraphNode  $end
     * @return float
     */
    public function distance(GraphNode $a, GraphNode $b) : float
    {
        $vectors = [0 => [], 1 => []];

        foreach ($this->axes as $axis) {
            $vectors[0][] = $a->get($axis);
            $vectors[1][] = $b->get($axis);
        }

        return $this->distanceFunction->compute(...$vectors);
    }

    /**
     * Find the K nearest neighbors of a given node. O(V logV)
     *
     * @param  \Rubix\Engine\Graph\GraphNode  $node
     * @param  int  $k
     * @param  bool  $self
     * @return \Rubix\Engine\Graph\ObjectIndex
     */
    public function findNearestNeighbors(GraphNode $node, int $k = 3, bool $self = false) : ObjectIndex
    {
        $neighbors = new SplPriorityQueue();
        $cluster = new ObjectIndex();

        foreach ($this->nodes as $neighbor) {
            if ($self !== true) {
                if ($neighbor->isSame($node)) {
                    continue;
                }
            }

            $distance = $this->distance($node, $neighbor);

            $neighbors->insert($neighbor, 1 - $distance);
        }

        while ($neighbors->valid()) {
            if ($cluster->count() >= $k) {
                break;
            }

            $neighbor = $neighbors->extract();

            $cluster->put($neighbor->id(), $neighbor);
        }

        return $cluster;
    }

    /**
     * Find the K farthest neighbors of a given node. O(V logV)
     *
     * @param  \Rubix\Engine\Graph\GraphNode  $node
     * @param  int  $k
     * @param  bool  $self
     * @return \Rubix\Engine\Graph\ObjectIndex
     */
    public function findFarthestNeighbors(GraphNode $node, int $k = 3, bool $self = false) : ObjectIndex
    {
        $neighbors = new SplPriorityQueue();
        $cluster = new ObjectIndex();

        foreach ($this->nodes as $neighbor) {
            if ($self !== true) {
                if ($neighbor->isSame($node)) {
                    continue;
                }
            }

            $distance = $this->distance($node, $neighbor);

            $neighbors->insert($neighbor, $distance);
        }

        while ($neighbors->valid()) {
            if ($cluster->count() >= $k) {
                break;
            }

            $neighbor = $neighbors->extract();

            $cluster->put($neighbor->id(), $neighbor);
        }

        return $cluster;
    }

    /**
     * Find the K nearest reachable neighbors of a given node.
     *
     * @param  \Rubix\Engine\Graph\GraphNode  $node
     * @param  int  $k
     * @param  bool  $self
     * @return \Rubix\Engine\Graph\ObjectIndex
     */
    public function findNearestReachableNeighbors(GraphNode $node, int $k = 3, bool $self = false) : ObjectIndex
    {
        $discovered = new SplObjectStorage();
        $neighbors = new SplPriorityQueue();
        $cluster = new ObjectIndex();
        $stack = [];

        $discovered->attach($node, null);

        array_push($stack, $node);

        while (!empty($stack)) {
            $current = array_pop($stack);

            foreach ($current->edges() as $edge) {
                $neighbor = $edge->node();

                if (!$discovered->contains($neighbor)) {
                    $discovered->attach($neighbor, $current);

                    if ($self !== true) {
                        if ($neighbor->isSame($node)) {
                            continue;
                        }
                    }

                    $distance = $this->distance($node, $neighbor);

                    $neighbors->insert($neighbor, 1 - $distance);

                    $stack[] = $neighbor;
                }
            }
        }

        while ($neighbors->valid()) {
            if ($cluster->count() >= $k) {
                break;
            }

            $neighbor = $neighbors->extract();

            $cluster->put($neighbor->id(), $neighbor);
        }

        return $cluster;
    }

    /**
     * Find the K farthest reachable neighbors of a given node.
     *
     * @param  \Rubix\Engine\Graph\GraphNode  $node
     * @param  int  $k
     * @return \Rubix\Engine\Graph\ObjectIndex
     */
    public function findFarthestReachableNeighbors(GraphNode $node, int $k = 3, bool $self = false) : ObjectIndex
    {
        $discovered = new SplObjectStorage();
        $neighbors = new SplPriorityQueue();
        $cluster = new ObjectIndex();
        $stack = [];

        array_push($stack, $node);

        while (!empty($stack)) {
            $current = array_pop($stack);

            foreach ($current->edges() as $edge) {
                $neighbor = $edge->node();

                if (!$discovered->contains($neighbor)) {
                    $discovered->attach($neighbor, $current);

                    if ($self !== true) {
                        if ($neighbor->isSame($node)) {
                            continue;
                        }
                    }

                    $distance = $this->distance($node, $neighbor);

                    $neighbors->insert($neighbor, $distance);

                    $stack[] = $neighbor;
                }
            }
        }

        while ($neighbors->valid()) {
            if ($cluster->count() >= $k) {
                break;
            }

            $neighbor = $neighbors->extract();

            $cluster->put($neighbor->id(), $neighbor);
        }

        return $cluster;
    }

    /**
     * Find a shortest smart path between a start node and an end node in a grid.
     * Uses a distance function to compute a heuristic that prioritizes the direction
     * of the traversal. Returns null if no path can be found. O(VlogV + ElogV)
     *
     * @param  \Rubix\Engine\GraphNode  $start
     * @param  \Rubix\Engine\GraphNode  $end
     * @return \Rubix\Engine\Path|null
     */
    public function findShortestSmartPath(GraphNode $start, GraphNode $end) : ?Path
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
                $distance = $discovered[$current]['distance'] + 1;

                if ($distance < $discovered[$edge->node()]['distance']) {
                    $discovered[$edge->node()] = [
                        'parent' => $current,
                        'distance' => $distance,
                    ];

                    $heuristic = $this->distance($current, $edge->node());

                    $queue->insert($edge->node(), -($distance + $heuristic));
                }
            }
        }

        return null;
    }

    /**
     * Find a shortest smart unsigned weighted path between a start node to an end
     * node in a grid. Uses a euclidian distance heuristic to prioritize the direction
     * of the traversal. Returns null if no path can be found. O(VlogV+ElogV)
     *
     * @param  \Rubix\Engine\GraphNode  $start
     * @param  \Rubix\Engine\GraphNode  $end
     * @param  string  $weight
     * @param  mixed  $default
     * @return \Rubix\Engine\Path|null
     */
    public function findShortestUnsignedWeightedSmartPath(GraphNode $start, GraphNode $end, string $weight, $default = INF) : ?Path
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

                    $heuristic = $this->distance($current, $edge->node());

                    $queue->insert($edge->node(), -($distance + $heuristic));
                }
            }
        }

        return null;
    }
}
