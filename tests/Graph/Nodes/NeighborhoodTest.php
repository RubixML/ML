<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Hypercube;
use Rubix\ML\Graph\Nodes\Neighborhood;
use PHPUnit\Framework\TestCase;

/**
 * @group Nodes
 * @covers \Rubix\ML\Graph\Nodes\Neighborhood
 */
class NeighborhoodTest extends TestCase
{
    protected const SAMPLES = [
        [5.0, 2.0, -3],
        [6.0, 4.0, -5],
    ];

    protected const LABELS = [
        22, 13,
    ];

    protected const MIN = [5.0, 2.0, -5];

    protected const MAX = [6.0, 4.0, -3];

    protected const BOX = [
        self::MIN, self::MAX,
    ];

    /**
     * @var Neighborhood
     */
    protected $node;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $dataset = Labeled::quick(self::SAMPLES, self::LABELS);

        $this->node = new Neighborhood($dataset, self::MIN, self::MAX);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Neighborhood::class, $this->node);
        $this->assertInstanceOf(Hypercube::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    /**
     * @test
     */
    public function terminate() : void
    {
        $node = Neighborhood::terminate(Labeled::quick(self::SAMPLES, self::LABELS));

        $this->assertInstanceOf(Neighborhood::class, $node);
        $this->assertInstanceOf(Labeled::class, $node->dataset());
        $this->assertEquals(self::BOX, iterator_to_array($node->sides()));
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
    public function sides() : void
    {
        $this->assertEquals(self::BOX, iterator_to_array($this->node->sides()));
    }
}
