<?php

use Rubix\Engine\Grid;
use Rubix\Engine\Path;
use PHPUnit\Framework\TestCase;

class GridTest extends TestCase
{
    protected $grid;

    public function setUp()
    {
        $this->grid = new Grid(['x', 'y']);

        for ($x = 0; $x < 10; $x++) {
            for ($y = 0; $y < 10; $y++) {
                $this->grid->insert(['x' => $x, 'y' => $y]);
            }
        }

        $directions = [[1, 0], [0, 1], [-1, 0], [0, -1]];

        foreach ($this->grid->nodes() as $node) {
            foreach ($directions as $direction) {
                $neighbor = $this->grid->nodes()
                    ->where('x', '==', $node->x + $direction[0])
                    ->where('y', '==', $node->y + $direction[1])
                    ->first();

                if (isset($neighbor)) {
                    $node->attach($neighbor, [
                        'difficulty' => rand(0, 10),
                    ]);
                }
            }
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
