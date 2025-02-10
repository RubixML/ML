<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Graph\Nodes;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Clique;
use Rubix\ML\Kernels\Distance\Euclidean;
use PHPUnit\Framework\TestCase;

#[Group('Nodes')]
#[CoversClass(Clique::class)]
class CliqueTest extends TestCase
{
    protected const array SAMPLES = [
        [5.0, 2.0, -3],
        [6.0, 4.0, -5],
    ];

    protected const array LABELS = [22, 13];

    protected const array CENTER = [5.5, 3.0, -4];

    protected const float RADIUS = 1.5;

    protected Clique $node;

    protected function setUp() : void
    {
        $dataset = Labeled::quick(samples: self::SAMPLES, labels: self::LABELS);

        $this->node = new Clique(
            dataset: $dataset,
            center: self::CENTER,
            radius: self::RADIUS
        );
    }

    public function testTerminate() : void
    {
        $dataset = Labeled::quick(samples: self::SAMPLES, labels: self::LABELS);

        $node = Clique::terminate(dataset: $dataset, kernel: new Euclidean());

        $this->assertEquals(self::CENTER, $node->center());
        $this->assertEquals(self::RADIUS, $node->radius());
    }

    public function testDataset() : void
    {
        $this->assertEquals(self::SAMPLES, $this->node->dataset()->samples());
        $this->assertEquals(self::LABELS, $this->node->dataset()->labels());
    }

    public function testCenter() : void
    {
        $this->assertEquals(self::CENTER, $this->node->center());
    }

    public function testRadius() : void
    {
        $this->assertEquals(self::RADIUS, $this->node->radius());
    }
}
