<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

/**
 * @group Nodes
 * @covers \Rubix\ML\Graph\Nodes\Split
 */
class SplitTest extends TestCase
{
    protected const COLUMN = 1;

    protected const VALUE = 3.0;

    protected const SAMPLES = [
        [5.0, 2.0, -3],
        [6.0, 4.0, -5],
    ];

    protected const LABELS = [22, 13];

    protected const IMPURITY = 400.0;

    protected const N = 4;

    /**
     * @var Split
     */
    protected $node;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $subsets = [
            Labeled::quick(self::SAMPLES, self::LABELS),
            Labeled::quick(self::SAMPLES, self::LABELS),
        ];

        $this->node = new Split(self::COLUMN, self::VALUE, $subsets, self::IMPURITY, self::N);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Split::class, $this->node);
        $this->assertInstanceOf(Decision::class, $this->node);
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
    public function subsets() : void
    {
        $expected = [
            Labeled::quick(self::SAMPLES, self::LABELS),
            Labeled::quick(self::SAMPLES, self::LABELS),
        ];

        $this->assertEquals($expected, $this->node->subsets());
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
    public function purityIncrease() : void
    {
        $this->node->attachLeft(new Split(2, 0.0, [Labeled::quick(), Labeled::quick()], 50.0, 1));
        $this->node->attachRight(new Split(4, -12.0, [Labeled::quick(), Labeled::quick()], 200.0, 3));

        $this->assertSame(237.5, $this->node->purityIncrease());
    }

    /**
     * @test
     */
    public function n() : void
    {
        $this->assertSame(self::N, $this->node->n());
    }

    /**
     * @test
     */
    public function cleanup() : void
    {
        $subsets = $this->node->subsets();

        $this->assertIsArray($subsets);
        $this->assertCount(2, $subsets);

        $this->node->cleanup();

        $this->expectException(RuntimeException::class);

        $this->node->subsets();
    }
}
