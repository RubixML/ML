<?php

namespace Rubix\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class ComparisonTest extends TestCase
{
    protected $node;

    protected $params;

    public function setUp()
    {
        $this->params = [1, 1000, [[[0, 2], [0, 3]], []], 0.9];

        $this->node = new Comparison(...$this->params);
    }

    public function test_build_node()
    {
        $this->assertInstanceOf(Comparison::class, $this->node);
        $this->assertInstanceOf(Split::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_get_score()
    {
        $this->assertEquals($this->params[3], $this->node->score());
    }

    public function test_get_n()
    {
        $this->assertEquals(2, $this->node->n());
    }
}
