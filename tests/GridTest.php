<?php

use Rubix\Engine\Grid;
use Rubix\Engine\Graph\Path;
use Rubix\Engine\DistanceFunctions\Manhattan;
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

    public function test_find_k_nearest_neighbors()
    {
        $node = $this->grid->nodes()->where('x', '===', 3)->where('y', '===', 7)->first();

        $neighbors = $this->grid->findKNearestNeighbors($node, 3);

        $this->assertEquals(3, count($neighbors));
        $this->assertEquals([2, 7], [$neighbors[0]->x, $neighbors[0]->y]);
        $this->assertEquals([3, 6], [$neighbors[1]->x, $neighbors[1]->y]);
        $this->assertEquals([3, 8], [$neighbors[2]->x, $neighbors[2]->y]);
    }

    public function test_find_k_farthest_neighbors()
    {
        $node = $this->grid->nodes()->where('x', '===', 3)->where('y', '===', 7)->first();

        $neighbors = $this->grid->findKFarthestNeighbors($node, 3);

        $this->assertEquals(3, count($neighbors));
        $this->assertEquals([9, 0], [$neighbors[0]->x, $neighbors[0]->y]);
        $this->assertEquals([8, 0], [$neighbors[1]->x, $neighbors[1]->y]);
        $this->assertEquals([9, 1], [$neighbors[2]->x, $neighbors[2]->y]);
    }

    public function test_find_k_nearest_attached_neighbors()
    {
        $node = $this->grid->nodes()->where('x', '===', 3)->where('y', '===', 7)->first();

        $neighbors = $this->grid->findKNearestAttachedNeighbors($node, 3);

        $this->assertEquals(3, count($neighbors));
        $this->assertEquals([4, 7], [$neighbors[0]->x, $neighbors[0]->y]);
        $this->assertEquals([3, 6], [$neighbors[1]->x, $neighbors[1]->y]);
        $this->assertEquals([2, 7], [$neighbors[2]->x, $neighbors[2]->y]);
    }

    public function test_find_k_farthest_attached_neighbors()
    {
        $node = $this->grid->nodes()->where('x', '===', 3)->where('y', '===', 7)->first();

        $neighbors = $this->grid->findKFarthestAttachedNeighbors($node, 3);

        $this->assertEquals(3, count($neighbors));
        $this->assertEquals([4, 7], [$neighbors[0]->x, $neighbors[0]->y]);
        $this->assertEquals([3, 6], [$neighbors[1]->x, $neighbors[1]->y]);
        $this->assertEquals([2, 7], [$neighbors[2]->x, $neighbors[2]->y]);
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
