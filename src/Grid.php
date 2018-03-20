<?php

namespace Rubix\Graph;

use Rubix\Graph\DistanceFunctions\Euclidean;
use Rubix\Graph\DistanceFunctions\DistanceFunction;
use InvalidArgumentException;
use SplObjectStorage;
use SplPriorityQueue;

class Grid extends Graph
{
    /**
     * The property names that represent the axis of the grid.
     *
     * @var array
     */
    protected $axis;

    /**
     * The distance function that describes the grid space.
     *
     * @var \Rubix\DistanceFunctions\DistanceFunction
     */
    protected $distanceFunction;

    /**
     * @param  array  $axis
     * @param  \Rubix\DistanceFunctions\DistanceFunction|null  $distance
     * @return void
     */
    public function __construct(array $axis, DistanceFunction $distanceFunction = null)
    {
        if (!isset($distanceFunction)) {
            $distanceFunction = new Euclidean();
        }

        $this->axis = $axis;
        $this->distanceFunction = $distanceFunction;

        parent::__construct();
    }

    /**
     * @return array
     */
    public function axis() : array
    {
        return $this->axis;
    }

    /**
     * Return the number of dimensions of the grid.
     *
     * @return int
     */
    public function dimensions() : int
    {
        return count($this->axis);
    }

    /**
     * Compute the distance between two given nodes.
     *
     * @param  \Rubix\Graph\GraphNode  $start
     * @param  \Rubix\Graph\GraphNode  $end
     * @return float
     */
    public function distance(GraphNode $start, GraphNode $end) : float
    {
        return $this->distanceFunction->compute($start, $end, $this->axis);
    }

    /**
     * Insert a node into the grid. O(1)
     *
     * @param  array  $properties
     * @throws \InvalidArgumentException
     * @return \Rubix\Graph\GraphGraphNode
     */
    public function insert(array $properties = []) : GraphNode
    {
        foreach ($this->axis as $axis) {
            if (!isset($properties[$axis])) {
                throw new InvalidArgumentException('Node must have a value set for all axis of the grid.');
            }
        }

        return parent::insert($properties);
    }

    /**
     * Find a shortest smart path between a start node and an end node in a grid.
     * Uses a euclidian distance heuristic to prioritize the direction of the
     * traversal. Returns null if no path can be found. O(VlogV + ElogV)
     *
     * @param  \Rubix\Graph\GraphNode  $start
     * @param  \Rubix\Graph\GraphNode  $end
     * @return \Rubix\Graph\Path|null
     */
    public function findShortestSmartPath(GraphNode $start, GraphNode $end) : ?Path
    {
        $discovered = new SplObjectStorage();
        $queue = new SplPriorityQueue();
        $path = new Path();

        foreach ($this->nodes()->all() as $node) {
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

            foreach ($current->edges()->all() as $edge) {
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
     * @param  \Rubix\Graph\GraphNode  $start
     * @param  \Rubix\Graph\GraphNode  $end
     * @param  string  $weight
     * @param  mixed  $default
     * @return \Rubix\Graph\Path|null
     */
    public function findShortestUnsignedWeightedSmartPath(GraphNode $start, GraphNode $end, string $weight, $default = INF) : ?Path
    {
        $discovered = new SplObjectStorage();
        $queue = new SplPriorityQueue();
        $path = new Path();

        foreach ($this->nodes()->all() as $node) {
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

            foreach ($current->edges()->all() as $edge) {
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
