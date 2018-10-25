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

    protected $params;

    public function setUp()
    {
        $this->params = [12, 0.9];

        $this->node = new Cell(...$this->params);
    }

    public function test_build_node()
    {
        $this->assertInstanceOf(Cell::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Leaf::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_get_n()
    {
        $this->assertEquals($this->params[0], $this->node->n());
    }

    public function test_get_score()
    {
        $this->assertEquals($this->params[1], $this->node->score());
    }
}
