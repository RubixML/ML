<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Coordinate;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class CoordinateTest extends TestCase
{
    protected const COLUMN = 1;
    protected const VALUE = 3.;

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
        $groups = [
            Labeled::quick([self::SAMPLES[0]], [self::LABELS[0]]),
            Labeled::quick([self::SAMPLES[1]], [self::LABELS[1]]),
        ];

        $node = new Coordinate(self::COLUMN, self::VALUE, $groups, self::MIN, self::MAX);

        $this->assertInstanceOf(Coordinate::class, $node);
        $this->assertInstanceOf(BinaryNode::class, $node);
        $this->assertInstanceOf(Node::class, $node);

        $this->assertEquals(self::BOX, iterator_to_array($node->sides()));
    }

    public function test_split()
    {
        $node = Coordinate::split(Labeled::quick(self::SAMPLES, self::LABELS));

        $this->assertEquals(self::BOX, iterator_to_array($node->sides()));
    }
}
