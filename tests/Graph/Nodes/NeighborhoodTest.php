<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Leaf;
use Rubix\ML\Graph\Nodes\Hypercube;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Graph\Nodes\Neighborhood;
use PHPUnit\Framework\TestCase;

class NeighborhoodTest extends TestCase
{
    protected const SAMPLES = [
        [5., 2., -3],
        [6., 4., -5],
    ];

    protected const LABELS = [
        22, 13,
    ];

    protected const MIN = [5., 2., -5];
    protected const MAX = [6., 4., -3];

    protected const BOX = [
        self::MIN, self::MAX,
    ];

    /**
     * @var \Rubix\ML\Graph\Nodes\Neighborhood
     */
    protected $node;

    public function setUp() : void
    {
        $this->node = new Neighborhood(self::SAMPLES, self::LABELS, self::MIN, self::MAX);
    }

    public function test_build_node() : void
    {
        $this->assertInstanceOf(Neighborhood::class, $this->node);
        $this->assertInstanceOf(Hypercube::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Leaf::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_terminate() : void
    {
        $node = Neighborhood::terminate(Labeled::quick(self::SAMPLES, self::LABELS));

        $this->assertEquals(self::SAMPLES, $node->samples());
        $this->assertEquals(self::LABELS, $node->labels());
        $this->assertEquals(self::BOX, iterator_to_array($node->sides()));
    }

    public function test_samples() : void
    {
        $this->assertEquals(self::SAMPLES, $this->node->samples());
    }

    public function test_labels() : void
    {
        $this->assertEquals(self::LABELS, $this->node->labels());
    }

    public function test_sides() : void
    {
        $this->assertEquals(self::BOX, iterator_to_array($this->node->sides()));
    }
}
