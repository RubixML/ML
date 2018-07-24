<?php

namespace Rubix\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class ComparisonTest extends TestCase
{
    protected $node;

    protected $params;

    public function setUp()
    {
        $this->params = [1, 1000, 0.9, []];

        $this->node = new Comparison(...$this->params);
    }

    public function test_build_node()
    {
        $this->assertInstanceOf(Comparison::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_get_index()
    {
        $this->assertEquals($this->params[0], $this->node->index());
    }

    public function test_get_value()
    {
        $this->assertEquals($this->params[1], $this->node->value());
    }

    public function test_get_score()
    {
        $this->assertEquals($this->params[2], $this->node->score());
    }

    public function test_get_groups()
    {
        $this->assertEquals($this->params[3], $this->node->groups());
    }
}
