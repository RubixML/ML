<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Graph\Nodes;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
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

    #[Test]
    public function split() : void
    {
        $node = Box::split(Labeled::quick(samples: self::SAMPLES, labels: self::LABELS));

        $this->assertEquals(self::BOX, iterator_to_array($node->sides()));
    }

    #[Test]
    public function column() : void
    {
        $this->assertSame(self::COLUMN, $this->node->column());
    }

    #[Test]
    public function value() : void
    {
        $this->assertSame(self::VALUE, $this->node->value());
    }

    #[Test]
    public function subsets() : void
    {
        $expected = [
            Labeled::quick(samples: [self::SAMPLES[0]], labels: [self::LABELS[0]]),
            Labeled::quick(samples: [self::SAMPLES[1]], labels: [self::LABELS[1]]),
        ];

        $this->assertEquals($expected, $this->node->subsets());
    }

    #[Test]
    public function sides() : void
    {
        $this->assertEquals(self::BOX, iterator_to_array($this->node->sides()));
    }

    #[Test]
    public function cleanup() : void
    {
        $this->node->cleanup();

        $this->expectException(RuntimeException::class);

        $this->node->subsets();
    }
}
