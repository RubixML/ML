<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Graph\Nodes;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Nodes')]
#[CoversClass(Split::class)]
class SplitTest extends TestCase
{
    protected const int COLUMN = 1;

    protected const float VALUE = 3.0;

    protected const array SAMPLES = [
        [5.0, 2.0, -3],
        [6.0, 4.0, -5],
    ];

    protected const array LABELS = [22, 13];

    protected const float IMPURITY = 400.0;

    protected const int N = 4;

    protected Split $node;

    protected function setUp() : void
    {
        $subsets = [
            Labeled::quick(samples: self::SAMPLES, labels: self::LABELS),
            Labeled::quick(samples: self::SAMPLES, labels: self::LABELS),
        ];

        $this->node = new Split(
            column: self::COLUMN,
            value: self::VALUE,
            subsets: $subsets,
            impurity: self::IMPURITY,
            n: self::N
        );
    }

    public function testColumn() : void
    {
        $this->assertSame(self::COLUMN, $this->node->column());
    }

    public function testValue() : void
    {
        $this->assertSame(self::VALUE, $this->node->value());
    }

    public function testSubsets() : void
    {
        $expected = [
            Labeled::quick(samples: self::SAMPLES, labels: self::LABELS),
            Labeled::quick(samples: self::SAMPLES, labels: self::LABELS),
        ];

        $this->assertEquals($expected, $this->node->subsets());
    }

    public function testImpurity() : void
    {
        $this->assertSame(self::IMPURITY, $this->node->impurity());
    }

    public function testPurityIncrease() : void
    {
        $this->node->attachLeft(new Split(
            column: 2,
            value: 0.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 50.0,
            n: 1
        ));
        $this->node->attachRight(new Split(
            column: 4,
            value: -12.0,
            subsets: [Labeled::quick(), Labeled::quick()],
            impurity: 200.0,
            n: 3
        ));

        $this->assertSame(237.5, $this->node->purityIncrease());
    }

    public function testN() : void
    {
        $this->assertSame(self::N, $this->node->n());
    }

    public function testCleanup() : void
    {
        $this->node->cleanup();

        $this->expectException(RuntimeException::class);

        $this->node->subsets();
    }
}
