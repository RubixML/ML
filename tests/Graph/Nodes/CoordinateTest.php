<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Graph\Nodes\Coordinate;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class CoordinateTest extends TestCase
{
    protected $node;

    protected $groups;

    public function setUp()
    {
        $this->groups = [Unlabeled::quick(), Unlabeled::quick()];

        $this->node = new Coordinate(5, 1, $this->groups);
    }

    public function test_build_node()
    {
        $this->assertInstanceOf(Coordinate::class, $this->node);
        $this->assertInstanceOf(Split::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }
}
