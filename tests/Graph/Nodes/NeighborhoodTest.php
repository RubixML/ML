<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Graph\Nodes;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Neighborhood;
use PHPUnit\Framework\TestCase;

#[Group('Nodes')]
#[CoversClass(Neighborhood::class)]
class NeighborhoodTest extends TestCase
{
    protected const array SAMPLES = [
        [5.0, 2.0, -3],
        [6.0, 4.0, -5],
    ];

    protected const array LABELS = [
        22, 13,
    ];

    protected const array MIN = [5.0, 2.0, -5];

    protected const array MAX = [6.0, 4.0, -3];

    protected const array BOX = [
        self::MIN, self::MAX,
    ];

    protected Neighborhood $node;

    protected function setUp() : void
    {
        $dataset = Labeled::quick(samples: self::SAMPLES, labels: self::LABELS);

        $this->node = new Neighborhood(dataset: $dataset, min: self::MIN, max: self::MAX);
    }

    public function testTerminate() : void
    {
        $node = Neighborhood::terminate(Labeled::quick(samples: self::SAMPLES, labels: self::LABELS));

        $this->assertEquals(self::BOX, iterator_to_array($node->sides()));
    }

    public function testDataset() : void
    {
        $this->assertEquals(self::SAMPLES, $this->node->dataset()->samples());
        $this->assertEquals(self::LABELS, $this->node->dataset()->labels());
    }

    public function testSides() : void
    {
        $this->assertEquals(self::BOX, iterator_to_array($this->node->sides()));
    }
}
