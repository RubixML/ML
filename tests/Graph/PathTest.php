<?php

use Rubix\ML\Graph\Path;
use Rubix\ML\Graph\GraphNode;
use PHPUnit\Framework\TestCase;

class PathTest extends TestCase
{
    protected $path;

    public function setUp()
    {
        $this->path = new Path([
            new GraphNode([
                'name' => 'San Francisco',
                'population' => 938493,
            ]),
            new GraphNode([
                'name' => 'Chicago',
                'population' => 847392,
            ]),
            new GraphNode([
                'name' => 'New York',
                'population' => 1847593,
            ]),
            new GraphNode([
                'name' => 'London',
                'population' => 958093,
            ])
        ]);
    }

    public function test_prepend_node()
    {
        $this->assertEquals('San Francisco', $this->path->first()->name);

        $this->path->prepend(new GraphNode([
            'name' => 'Paris',
            'population' => 453234,
        ]));

        $this->assertEquals('Paris', $this->path->first()->name);
    }

    public function test_append_node()
    {
        $this->assertEquals('London', $this->path->last()->name);

        $this->path->append(new GraphNode([
            'name' => 'Paris',
            'population' => 453234,
        ]));

        $this->assertEquals('Paris', $this->path->last()->name);
    }

    public function test_count_nodes_in_path()
    {
        $this->assertEquals(4, $this->path->count());
    }

    public function test_first_node()
    {
        $this->assertEquals('San Francisco', $this->path->first()->name);
    }

    public function test_current_next_and_previous_nodes()
    {
        $this->path->rewind();

        $this->assertEquals('San Francisco', $this->path->current()->name);
        $this->assertEquals('Chicago', $this->path->next()->name);
        $this->assertEquals('San Francisco', $this->path->prev()->name);
    }

    public function test_last_node()
    {
        $this->assertEquals('London', $this->path->last()->name);
    }

    public function test_return_all_nodes()
    {
        $nodes = $this->path->all();

        $this->assertEquals('San Francisco', $nodes[0]->name);
        $this->assertEquals('Chicago', $nodes[1]->name);
        $this->assertEquals('New York', $nodes[2]->name);
        $this->assertEquals('London', $nodes[3]->name);
    }
}
