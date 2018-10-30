<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Cell;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Leaf;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class CellTest extends TestCase
{
    protected $node;

    public function setUp()
    {
        $this->node = new Cell(12.9, 3);
    }

    public function test_build_node()
    {
        $this->assertInstanceOf(Cell::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Leaf::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_get_depth()
    {
        $this->assertEquals(12.9, $this->node->depth());
    }

    public function test_get_n()
    {
        $this->assertEquals(3, $this->node->n());
    }
}
