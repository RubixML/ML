<?php

use Rubix\Engine\Grid;
use Rubix\Engine\Graph\Path;
use Rubix\Engine\Graph\DistanceFunctions\Manhattan;
use PHPUnit\Framework\TestCase;

class GridTest extends TestCase
{
    protected $grid;

    public function setUp()
    {
        $this->grid = new Grid(['x','y'], new Manhattan());

        for ($x = 0; $x < 10; $x++) {
            for ($y = 0; $y < 10; $y++) {
                $this->grid->insert(['x' => $x, 'y' => $y]);
            }
        }

        $directions = [[1, 0], [0, 1], [-1, 0], [0, -1]];

        foreach ($this->grid->nodes() as $node) {
            foreach ($directions as $direction) {
                $neighbor = $this->grid->nodes()
                    ->where('x', '===', $node->x + $direction[0])
                    ->where('y', '===', $node->y + $direction[1])
                    ->first();

                if (isset($neighbor)) {
                    $node->attach($neighbor, [
                        'difficulty' => rand(0, 10),
                    ]);
                }
            }
        }
    }

    public function test_compute_distance()
    {
        $start = $this->grid->nodes()->where('x', '===', 3)->where('y', '===', 7)->first();
        $end = $this->grid->nodes()->where('x', '===', 5)->where('y', '===', 2)->first();

        $this->assertEquals(7.0, round($this->grid->distance($start, $end), 2));
    }

    public function test_find_nearest_neighbors()
    {
        $node = $this->grid->nodes()->where('x', '===', 3)->where('y', '===', 7)->first();

        $neighbors = $this->grid->findNearestNeighbors($node, 3);

        $this->assertEquals(3, $neighbors->count());

        foreach ($neighbors as $neighbor) {
            $this->assertTrue(in_array([$neighbor->x, $neighbor->y], [[2, 7], [3, 6], [3, 8]]));
        }
    }

    public function test_find_farthest_neighbors()
    {
        $node = $this->grid->nodes()->where('x', '===', 3)->where('y', '===', 7)->first();

        $neighbors = $this->grid->findFarthestNeighbors($node, 3);

        $this->assertEquals(3, $neighbors->count());

        foreach ($neighbors as $neighbor) {
            $this->assertTrue(in_array([$neighbor->x, $neighbor->y], [[9, 0], [8, 0], [9, 1]]));
        }
    }

    public function test_find_nearest_reachable_neighbors()
    {
        $node = $this->grid->nodes()->where('x', '===', 3)->where('y', '===', 7)->first();

        $neighbors = $this->grid->findNearestReachableNeighbors($node, 3);

        $this->assertEquals(3, $neighbors->count());

        foreach ($neighbors as $neighbor) {
            $this->assertTrue(in_array([$neighbor->x, $neighbor->y], [[3, 8], [3, 6], [4, 7]]));
        }
    }

    public function test_find_farthest_reachable_neighbors()
    {
        $node = $this->grid->nodes()->where('x', '===', 3)->where('y', '===', 7)->first();

        $neighbors = $this->grid->findFarthestReachableNeighbors($node, 3);

        $this->assertEquals(3, $neighbors->count());

        foreach ($neighbors as $neighbor) {
            $this->assertTrue(in_array([$neighbor->x, $neighbor->y], [[9, 0], [8, 0], [9, 1]]));
        }
    }

    public function test_find_shortest_smart_path()
    {
        $path = $this->grid->findShortestSmartPath($this->grid->find(1), $this->grid->find(50));

        $this->assertTrue($path instanceof Path);
        $this->assertEquals(14, $path->count());
    }

    public function test_find_shortest_smart_unsigned_weighted_path()
    {
        $path = $this->grid->findShortestUnsignedWeightedSmartPath($this->grid->find(1), $this->grid->find(50), 'difficulty');

        $this->assertTrue($path instanceof Path);
    }
}
