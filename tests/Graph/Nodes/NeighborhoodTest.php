<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Leaf;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Graph\Nodes\Neighborhood;
use PHPUnit\Framework\TestCase;

class NeighborhoodTest extends TestCase
{
    protected $node;

    protected $params;

    public function setUp()
    {
        $this->params = [[[2, 5], [7, 9]], ['yes', 'no']];

        $this->node = new Neighborhood(...$this->params);
    }

    public function test_build_node()
    {
        $this->assertInstanceOf(Neighborhood::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Leaf::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_get_samples()
    {
        $this->assertEquals($this->params[0], $this->node->samples());
    }

    public function test_get_labels()
    {
        $this->assertEquals($this->params[1], $this->node->labels());
    }
}
