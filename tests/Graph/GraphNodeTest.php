<?php

use Rubix\ML\Graph\GraphNode;
use PHPUnit\Framework\TestCase;

class GraphNodeTest extends TestCase
{
    protected $node;

    public function setUp()
    {
        GraphNode::resetCounter();
        
        $this->node = new GraphNode([
            'color' => 'orange',
        ]);
    }

    public function test_create_node()
    {
        $this->assertTrue($this->node instanceof GraphNode);
    }

    public function test_get_id()
    {
        $this->assertEquals(1, $this->node->id());
    }

    public function test_attach_and_detach_edges()
    {
        $node = new GraphNode([
            'color' => 'brown',
        ]);

        $this->node->attach($node, [
            'weight' => 7,
        ]);

        $this->assertNotNull($this->node->edges()->get(2));
        $this->assertEquals('brown', $this->node->edges()->first()->node()->color);
        $this->assertEquals(7, $this->node->edges()->first()->weight);

        $this->node->detach($node);

        $this->assertEquals(0, $this->node->edges()->count());

        $this->assertNull($this->node->edges()->get(2));
    }
}
