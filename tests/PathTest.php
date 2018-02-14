<?php

use Rubix\Engine\Path;
use Rubix\Engine\Node;
use PHPUnit\Framework\TestCase;

class PathTest extends TestCase
{
    protected $path;

    public function setUp()
    {
        $this->path = new Path([
            new Node(1, [
                'name' => 'San Francisco',
                'population' => 938493,
            ]),
            new Node(2, [
                'name' => 'Chicago',
                'population' => 847392,
            ]),
            new Node(3, [
                'name' => 'New York',
                'population' => 1847593,
            ]),
            new Node(4, [
                'name' => 'London',
                'population' => 958093,
            ])
        ]);
    }

    public function test_prepend_node()
    {
        $this->assertEquals('San Francisco', $this->path->first()->name);

        $this->path->prepend(new Node(5, [
            'name' => 'Paris',
            'population' => 453234,
        ]));

        $this->assertEquals('Paris', $this->path->first()->name);
    }

    public function test_append_node()
    {
        $this->assertEquals('London', $this->path->last()->name);

        $this->path->append(new Node(5, [
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
