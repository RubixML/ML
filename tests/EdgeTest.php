<?php

use Rubix\Engine\GraphNode;
use Rubix\Engine\Edge;
use PHPUnit\Framework\TestCase;

class EdgeTest extends TestCase
{
    protected $edge;

    public function setUp()
    {
        $node = new GraphNode(1, [
            'name' => 'Programming',
        ]);

        $this->edge = new Edge($node, [
            'weight' => 10,
            'since' => '1997',
        ]);
    }

    public function test_create_edge()
    {
        $this->assertTrue($this->edge instanceof Edge);
        $this->assertEquals(10, $this->edge->weight);
        $this->assertEquals('1997', $this->edge->get('since'));
    }


    public function test_get_node()
    {
        $this->assertEquals('Programming', $this->edge->node()->name);
    }
}
