<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Leaf;
use Rubix\ML\Graph\Nodes\Hypercube;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Graph\Nodes\Neighborhood;
use PHPUnit\Framework\TestCase;

class NeighborhoodTest extends TestCase
{
    protected const SAMPLES = [
        [5., 2., -3],
        [6., 4., -5],
    ];

    protected const LABELS = [
        22, 13,
    ];

    protected const MIN = [5., 2., -5];
    protected const MAX = [6., 4., -3];

    protected const BOX = [
        self::MIN, self::MAX,
    ];

    public function test_build_node()
    {
        $node = new Neighborhood(self::SAMPLES, self::LABELS, self::MIN, self::MAX);

        $this->assertInstanceOf(Neighborhood::class, $node);
        $this->assertInstanceOf(Hypercube::class, $node);
        $this->assertInstanceOf(BinaryNode::class, $node);
        $this->assertInstanceOf(Leaf::class, $node);
        $this->assertInstanceOf(Node::class, $node);

        $this->assertEquals(self::SAMPLES, $node->samples());
        $this->assertEquals(self::LABELS, $node->labels());
        $this->assertEquals(self::BOX, iterator_to_array($node->sides()));
    }

    public function test_terminate()
    {
        $node = Neighborhood::terminate(Labeled::quick(self::SAMPLES, self::LABELS));

        $this->assertEquals(self::SAMPLES, $node->samples());
        $this->assertEquals(self::LABELS, $node->labels());
        $this->assertEquals(self::BOX, iterator_to_array($node->sides()));
    }
}
