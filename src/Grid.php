<?php

namespace Rubix\Engine;

use SplObjectStorage;
use SplPriorityQueue;

class Grid extends Graph
{
    /**
     * The properties that represent axis of the grid.
     *
     * @var array
     */
    protected $labels = [
        //
    ];

    /**
     * @param  array  $labels
     * @return void
     */
    public function __construct(array $labels = ['x', 'y'])
    {
        $this->labels = $labels;

        parent::__construct();
    }

    /**
     * Return the axis labels.
     *
     * @return int
     */
    public function labels() : array
    {
        return $this->labels;
    }

    /**
     * Return the number of dimensions of the grid.
     *
     * @return int
     */
    public function dimensions() : int
    {
        return count($this->labels);
    }

    /**
     * Find a shortest smart path between a start node and an end node in a grid.
     * Uses a euclidian distance heuristic to prioritize the direction of the
     * traversal. Returns null if no path can be found. O(VlogV+ElogV)
     *
     * @param  \Rubix\Engine\Node  $start
     * @param  \Rubix\Engine\Node  $end
     * @return \Rubix\Engine\Path|null
     */
    public function findShortestSmartPath(Node $start, Node $end) : ?Path
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

                    $heuristic = sqrt(array_reduce($this->labels, function ($carry, $label) use ($current, $edge) {
                        return $carry += pow($edge->node()->get($label, INF) - $current->get($label, INF), 2);
                    }, 0));

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
     * @param  \Rubix\Engine\Node  $start
     * @param  \Rubix\Engine\Node  $end
     * @param  string  $weight
     * @param  mixed  $default
     * @return \Rubix\Engine\Path|null
     */
    public function findShortestUnsignedWeightedSmartPath(Node $start, Node $end, string $weight, $default = INF) : ?Path
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

                    $heuristic = sqrt(array_reduce($this->labels, function ($carry, $label) use ($current, $edge) {
                        return $carry += pow($edge->node()->get($label, INF) - $current->get($label, INF), 2);
                    }, 0));

                    $queue->insert($edge->node(), -($distance + $heuristic));
                }
            }
        }

        return null;
    }
}
