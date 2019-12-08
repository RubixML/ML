<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Decision;
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

    protected const LABELS = [22, 13];

    protected const IMPURITY = 400.;

    /**
     * @var \Rubix\ML\Graph\Nodes\Comparison
     */
    protected $node;

    public function setUp() : void
    {
        $groups = [
            Labeled::quick([self::SAMPLES[0]], [self::LABELS[0]]),
            Labeled::quick([self::SAMPLES[1]], [self::LABELS[1]]),
        ];

        $this->node = new Comparison(self::COLUMN, self::VALUE, $groups, self::IMPURITY);
    }

    public function test_build_node() : void
    {
        $this->assertInstanceOf(Comparison::class, $this->node);
        $this->assertInstanceOf(Decision::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_column() : void
    {
        $this->assertSame(self::COLUMN, $this->node->column());
    }

    public function test_value() : void
    {
        $this->assertSame(self::VALUE, $this->node->value());
    }

    public function test_groups() : void
    {
        $expected = [
            Labeled::quick([self::SAMPLES[0]], [self::LABELS[0]]),
            Labeled::quick([self::SAMPLES[1]], [self::LABELS[1]]),
        ];

        $this->assertEquals($expected, $this->node->groups());
    }

    public function test_impurity() : void
    {
        $this->assertSame(self::IMPURITY, $this->node->impurity());
    }

    public function test_n() : void
    {
        $this->assertSame(2, $this->node->n());
    }
}
