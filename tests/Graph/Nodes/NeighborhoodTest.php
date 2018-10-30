<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
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
        $dataset = Labeled::quick([[7], [3]], ['cat', 'dog']);

        $this->node = new Neighborhood($dataset);
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
        $this->assertEquals([[7], [3]], $this->node->samples());
    }

    public function test_get_labels()
    {
        $this->assertEquals(['cat', 'dog'], $this->node->labels());
    }
}
