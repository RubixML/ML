<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Ball;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Graph\Nodes\Hypersphere;
use Rubix\ML\Kernels\Distance\Euclidean;
use PHPUnit\Framework\TestCase;

class BallTest extends TestCase
{
    protected const SAMPLES = [
        [5., 2., -3],
        [6., 4., -5],
    ];

    protected const LABELS = [22, 13];

    protected const CENTER = [5.5, 3., -4];

    protected const RADIUS = 1.5;

    /**
     * @var \Rubix\ML\Graph\Nodes\Ball
     */
    protected $node;

    public function setUp() : void
    {
        $groups = [
            Labeled::quick([self::SAMPLES[0]], [self::LABELS[0]]),
            Labeled::quick([self::SAMPLES[1]], [self::LABELS[1]]),
        ];

        $this->node = new Ball(self::CENTER, self::RADIUS, $groups);
    }

    public function test_build_node() : void
    {
        $this->assertInstanceOf(Ball::class, $this->node);
        $this->assertInstanceOf(Hypersphere::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_split() : void
    {
        $dataset = Labeled::quick(self::SAMPLES, self::LABELS);

        $node = Ball::split($dataset, new Euclidean());

        $this->assertEquals(self::CENTER, $node->center());
        $this->assertEquals(self::RADIUS, $node->radius());
    }

    public function test_center() : void
    {
        $this->assertSame(self::CENTER, $this->node->center());
    }

    public function test_radius() : void
    {
        $this->assertSame(self::RADIUS, $this->node->radius());
    }

    public function test_groups() : void
    {
        $expected = [
            Labeled::quick([self::SAMPLES[0]], [self::LABELS[0]]),
            Labeled::quick([self::SAMPLES[1]], [self::LABELS[1]]),
        ];

        $this->assertEquals($expected, $this->node->groups());
    }
}
