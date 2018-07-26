<?php

namespace Rubix\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Graph\Nodes\Coordinate;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class CoordinateTest extends TestCase
{
    protected $node;

    protected $params;

    public function setUp()
    {
        $this->params = [1, 1000, [[[0, 2], [0, 3]], []]];

        $this->node = new Coordinate(...$this->params);
    }

    public function test_build_node()
    {
        $this->assertInstanceOf(Coordinate::class, $this->node);
        $this->assertInstanceOf(Split::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }
}
