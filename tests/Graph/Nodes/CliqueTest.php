<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Clique;
use Rubix\ML\Graph\Nodes\Hypersphere;
use Rubix\ML\Kernels\Distance\Euclidean;
use PHPUnit\Framework\TestCase;

/**
 * @group Nodes
 * @covers \Rubix\ML\Graph\Nodes\Clique
 */
class CliqueTest extends TestCase
{
    protected const SAMPLES = [
        [5.0, 2.0, -3],
        [6.0, 4.0, -5],
    ];

    protected const LABELS = [22, 13];

    protected const CENTER = [5.5, 3.0, -4];

    protected const RADIUS = 1.5;

    /**
     * @var \Rubix\ML\Graph\Nodes\Clique
     */
    protected $node;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $dataset = Labeled::quick(self::SAMPLES, self::LABELS);

        $this->node = new Clique($dataset, self::CENTER, self::RADIUS);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Clique::class, $this->node);
        $this->assertInstanceOf(Hypersphere::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    /**
     * @test
     */
    public function terminate() : void
    {
        $dataset = Labeled::quick(self::SAMPLES, self::LABELS);

        $node = Clique::terminate($dataset, new Euclidean());

        $this->assertInstanceOf(Clique::class, $node);
        $this->assertInstanceOf(Labeled::class, $node->dataset());
        $this->assertEquals(self::CENTER, $node->center());
        $this->assertEquals(self::RADIUS, $node->radius());
    }

    /**
     * @test
     */
    public function dataset() : void
    {
        $this->assertInstanceOf(Labeled::class, $this->node->dataset());
        $this->assertEquals(self::SAMPLES, $this->node->dataset()->samples());
        $this->assertEquals(self::LABELS, $this->node->dataset()->labels());
    }

    /**
     * @test
     */
    public function center() : void
    {
        $this->assertEquals(self::CENTER, $this->node->center());
    }

    /**
     * @test
     */
    public function radius() : void
    {
        $this->assertEquals(self::RADIUS, $this->node->radius());
    }
}
