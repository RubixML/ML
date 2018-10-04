<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Graph\Nodes\Isolator;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class IsolatorTest extends TestCase
{
    protected $node;

    protected $params;

    public function setUp()
    {
        $this->params = [1, 1000, [[[0, 2], [0, 3]], []]];

        $this->node = new Isolator(...$this->params);
    }

    public function test_build_node()
    {
        $this->assertInstanceOf(Isolator::class, $this->node);
        $this->assertInstanceOf(Split::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }
}
