<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Leaf;
use Rubix\ML\Graph\Nodes\Best;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class BestTest extends TestCase
{
    protected $node;

    protected $params;

    public function setUp()
    {
        $this->node = new Best('cat', ['cat' => 0.9, 'pencil' => 0.1], 14.1, 6);
    }

    public function test_build_node()
    {
        $this->assertInstanceOf(Best::class, $this->node);
        $this->assertInstanceOf(Decision::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Leaf::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_get_outcome()
    {
        $this->assertEquals('cat', $this->node->outcome());
    }

    public function test_get_probabilities()
    {
        $this->assertEquals(['cat' => 0.9, 'pencil' => 0.1], $this->node->probabilities());
    }

    public function test_get_impurity()
    {
        $this->assertEquals(14.1, $this->node->impurity());
    }

    public function test_get_n()
    {
        $this->assertEquals(6, $this->node->n());
    }
}
