<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Graph\Nodes\Hypersphere;
use Rubix\ML\Graph\Nodes\VantagePoint;
use Rubix\ML\Kernels\Distance\Euclidean;
use PHPUnit\Framework\TestCase;

/**
 * @group Nodes
 * @covers \Rubix\ML\Graph\Nodes\VantagePoint
 */
class VantagePointTest extends TestCase
{
    protected const SAMPLES = [
        [5.0, 2.0, -3],
        [6.0, 4.0, -5],
    ];

    protected const LABELS = [22, 13];

    protected const CENTER = [5.5, 3.0, -4];

    protected const RADIUS = 1.5;

    /**
     * @var VantagePoint
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

        $this->node = new VantagePoint(self::CENTER, self::RADIUS, $groups);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(VantagePoint::class, $this->node);
        $this->assertInstanceOf(Hypersphere::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    /**
     * @test
     */
    public function split() : void
    {
        $dataset = Labeled::quick(self::SAMPLES, self::LABELS);

        $node = VantagePoint::split($dataset, new Euclidean());

        $this->assertEquals(self::CENTER, $node->center());
        $this->assertEquals(self::RADIUS, $node->radius());
    }

    /**
     * @test
     */
    public function center() : void
    {
        $this->assertSame(self::CENTER, $this->node->center());
    }

    /**
     * @test
     */
    public function radius() : void
    {
        $this->assertSame(self::RADIUS, $this->node->radius());
    }

    /**
     * @test
     */
    public function subsets() : void
    {
        $expected = [
            Labeled::quick([self::SAMPLES[0]], [self::LABELS[0]]),
            Labeled::quick([self::SAMPLES[1]], [self::LABELS[1]]),
        ];

        $this->assertEquals($expected, $this->node->subsets());
    }
}
