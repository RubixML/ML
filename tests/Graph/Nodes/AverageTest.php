<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Leaf;
use Rubix\ML\Graph\Nodes\Average;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class AverageTest extends TestCase
{
    protected $node;

    protected $params;

    public function setUp()
    {
        $this->node = new Average(44.21, 6., 3);
    }

    public function test_build_node()
    {
        $this->assertInstanceOf(Average::class, $this->node);
        $this->assertInstanceOf(Decision::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Leaf::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_get_outcome()
    {
        $this->assertEquals(44.21, $this->node->outcome());
    }

    public function test_get_impurity()
    {
        $this->assertEquals(6., $this->node->impurity());
    }

    public function test_get_n()
    {
        $this->assertEquals(3, $this->node->n());
    }
}
