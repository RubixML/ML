<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class ComparisonTest extends TestCase
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

    protected const IMPURITY = 400.;
    protected const N = 2;

    public function test_build_node()
    {
        $groups = [
            Labeled::quick([self::SAMPLES[0]], [self::LABELS[0]]),
            Labeled::quick([self::SAMPLES[1]], [self::LABELS[1]]),
        ];

        $node = new Comparison(self::COLUMN, self::VALUE, $groups, self::IMPURITY);

        $this->assertInstanceOf(Comparison::class, $node);
        $this->assertInstanceOf(BinaryNode::class, $node);
        $this->assertInstanceOf(Node::class, $node);

        $this->assertEquals(self::COLUMN, $node->column());
        $this->assertEquals(self::VALUE, $node->value());
        $this->assertEquals($groups, $node->groups());
        $this->assertEquals(self::IMPURITY, $node->impurity());
        $this->assertEquals(self::N, $node->n());
    }
}
