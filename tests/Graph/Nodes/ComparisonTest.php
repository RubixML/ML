<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class ComparisonTest extends TestCase
{
    protected $node;

    protected $groups;

    public function setUp()
    {
        $this->groups = [Unlabeled::quick(), Unlabeled::quick()];

        $this->node = new Comparison(3, -41, $this->groups, 400.);
    }

    public function test_build_node()
    {
        $this->assertInstanceOf(Comparison::class, $this->node);
        $this->assertInstanceOf(Split::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_get_impurity()
    {
        $this->assertEquals(400., $this->node->impurity());
    }

    public function test_get_n()
    {
        $this->assertEquals(0, $this->node->n());
    }
}
