<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Graph\Nodes;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Graph\Nodes\Box;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Nodes')]
#[CoversClass(Box::class)]
class BoxTest extends TestCase
{
    protected const int COLUMN = 1;

    protected const float VALUE = 3.;

    protected const array SAMPLES = [
        [5., 2., -3],
        [6., 4., -5],
    ];

    protected const array LABELS = [22, 13];

    protected const array MIN = [5., 2., -5];

    protected const array MAX = [6., 4., -3];

    protected const array BOX = [
        self::MIN, self::MAX,
    ];

    protected Box $node;

    protected function setUp() : void
    {
        $subsets = [
            Labeled::quick(samples: [self::SAMPLES[0]], labels: [self::LABELS[0]]),
            Labeled::quick(samples: [self::SAMPLES[1]], labels: [self::LABELS[1]]),
        ];

        $this->node = new Box(
            column: self::COLUMN,
            value: self::VALUE,
            subsets: $subsets,
            min: self::MIN,
            max: self::MAX
        );
    }

    public function testSplit() : void
    {
        $node = Box::split(Labeled::quick(samples: self::SAMPLES, labels: self::LABELS));

        $this->assertEquals(self::BOX, iterator_to_array($node->sides()));
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
            Labeled::quick(samples: [self::SAMPLES[0]], labels: [self::LABELS[0]]),
            Labeled::quick(samples: [self::SAMPLES[1]], labels: [self::LABELS[1]]),
        ];

        $this->assertEquals($expected, $this->node->subsets());
    }

    public function testSides() : void
    {
        $this->assertEquals(self::BOX, iterator_to_array($this->node->sides()));
    }

    public function testCleanup() : void
    {
        $this->node->cleanup();

        $this->expectException(RuntimeException::class);

        $this->node->subsets();
    }
}
