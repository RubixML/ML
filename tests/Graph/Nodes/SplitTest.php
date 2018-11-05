<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class SplitTest extends TestCase
{
    protected $node;

    protected $groups;

    public function setUp()
    {
        $this->groups = [Unlabeled::quick(), Unlabeled::quick()];

        $this->node = new Split(3, 45.5, $this->groups);
    }

    public function test_build_node()
    {
        $this->assertInstanceOf(Split::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_get_column()
    {
        $this->assertEquals(3, $this->node->column());
    }

    public function test_get_value()
    {
        $this->assertEquals(45.5, $this->node->value());
    }

    public function test_get_groups()
    {
        $this->assertEquals($this->groups, $this->node->groups());
    }
}
