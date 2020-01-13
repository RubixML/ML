<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

/**
 * @group Nodes
 * @covers \Rubix\ML\Graph\Nodes\Comparison
 */
class ComparisonTest extends TestCase
{
    protected const COLUMN = 1;
    protected const VALUE = 3.0;

    protected const SAMPLES = [
        [5.0, 2.0, -3],
        [6.0, 4.0, -5],
    ];

    protected const LABELS = [22, 13];

    protected const IMPURITY = 400.0;

    /**
     * @var \Rubix\ML\Graph\Nodes\Comparison
     */
    protected $node;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $groups = [
            Labeled::quick([self::SAMPLES[0]], [self::LABELS[0]]),
            Labeled::quick([self::SAMPLES[1]], [self::LABELS[1]]),
        ];

        $this->node = new Comparison(self::COLUMN, self::VALUE, $groups, self::IMPURITY);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Comparison::class, $this->node);
        $this->assertInstanceOf(Decision::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    /**
     * @test
     */
    public function column() : void
    {
        $this->assertSame(self::COLUMN, $this->node->column());
    }

    /**
     * @test
     */
    public function value() : void
    {
        $this->assertSame(self::VALUE, $this->node->value());
    }

    /**
     * @test
     */
    public function groups() : void
    {
        $expected = [
            Labeled::quick([self::SAMPLES[0]], [self::LABELS[0]]),
            Labeled::quick([self::SAMPLES[1]], [self::LABELS[1]]),
        ];

        $this->assertEquals($expected, $this->node->groups());
    }

    /**
     * @test
     */
    public function impurity() : void
    {
        $this->assertSame(self::IMPURITY, $this->node->impurity());
    }

    /**
     * @test
     */
    public function n() : void
    {
        $this->assertSame(2, $this->node->n());
    }
}
