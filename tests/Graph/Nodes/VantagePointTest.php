<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Graph\Nodes;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\VantagePoint;
use Rubix\ML\Kernels\Distance\Euclidean;
use PHPUnit\Framework\TestCase;

#[Group('Nodes')]
#[CoversClass(VantagePoint::class)]
class VantagePointTest extends TestCase
{
    protected const array SAMPLES = [
        [5.0, 2.0, -3],
        [6.0, 4.0, -5],
    ];

    protected const array LABELS = [22, 13];

    protected const array CENTER = [5.5, 3.0, -4];

    protected const float RADIUS = 1.5;

    protected VantagePoint $node;

    protected function setUp() : void
    {
        $groups = [
            Labeled::quick(samples: [self::SAMPLES[0]], labels: [self::LABELS[0]]),
            Labeled::quick(samples: [self::SAMPLES[1]], labels: [self::LABELS[1]]),
        ];

        $this->node = new VantagePoint(
            center:self::CENTER,
            radius: self::RADIUS,
            subsets: $groups
        );
    }

    public function testSplit() : void
    {
        $dataset = Labeled::quick(samples: self::SAMPLES, labels: self::LABELS);

        $node = VantagePoint::split(dataset: $dataset, kernel: new Euclidean());

        $this->assertEquals(self::CENTER, $node->center());
        $this->assertEquals(self::RADIUS, $node->radius());
    }

    public function testCenter() : void
    {
        $this->assertSame(self::CENTER, $this->node->center());
    }

    public function testRadius() : void
    {
        $this->assertSame(self::RADIUS, $this->node->radius());
    }

    public function testSubsets() : void
    {
        $expected = [
            Labeled::quick(samples: [self::SAMPLES[0]], labels: [self::LABELS[0]]),
            Labeled::quick(samples: [self::SAMPLES[1]], labels: [self::LABELS[1]]),
        ];

        $this->assertEquals($expected, $this->node->subsets());
    }
}
