<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Graph\Nodes;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Graph\Nodes\Depth;
use Rubix\ML\Datasets\Unlabeled;
use PHPUnit\Framework\TestCase;

#[Group('Nodes')]
#[CoversClass(Depth::class)]
class DepthTest extends TestCase
{
    protected const array SAMPLES = [
        [5.0, 2.0, -3],
        [6.0, 4.0, -5],
        [-0.01, 0.1, -7],
    ];

    protected const int DEPTH = 8;

    protected const float C = 8.207392357589622;

    protected Depth $node;

    protected function setUp() : void
    {
        $this->node = new Depth(self::C);
    }

    public function testC() : void
    {
        $this->assertEquals(3.748880484475505, Depth::c(10));
        $this->assertEquals(8.364671030072245, Depth::c(100));
        $this->assertEquals(12.969940887100174, Depth::c(1000));
        $this->assertEquals(17.575112063754766, Depth::c(10000));
        $this->assertEquals(22.180282259643523, Depth::c(100000));
    }

    public function testTerminate() : void
    {
        $dataset = Unlabeled::quick(samples: self::SAMPLES);

        $node = Depth::terminate(dataset: $dataset, depth: self::DEPTH);

        $this->assertEquals(self::C, $node->depth());
    }

    public function testDepth() : void
    {
        $this->assertEquals(self::C, $this->node->depth());
    }
}
