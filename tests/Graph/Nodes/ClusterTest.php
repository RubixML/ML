<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Leaf;
use Rubix\ML\Graph\Nodes\Cluster;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Graph\Nodes\Hypersphere;
use Rubix\ML\Kernels\Distance\Euclidean;
use PHPUnit\Framework\TestCase;

class ClusterTest extends TestCase
{
    protected const SAMPLES = [
        [5., 2., -3],
        [6., 4., -5],
    ];

    protected const LABELS = [22, 13];

    protected const CENTER = [5.5, 3., -4];

    protected const RADIUS = 1.5;

    /**
     * @var \Rubix\ML\Graph\Nodes\Cluster
     */
    protected $node;

    public function setUp() : void
    {
        $this->node = new Cluster(self::SAMPLES, self::LABELS, self::CENTER, self::RADIUS);
    }

    public function test_build_node() : void
    {
        $this->assertInstanceOf(Cluster::class, $this->node);
        $this->assertInstanceOf(Hypersphere::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Leaf::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_terminate() : void
    {
        $dataset = Labeled::quick(self::SAMPLES, self::LABELS);

        $node = Cluster::terminate($dataset, new Euclidean());

        $this->assertEquals(self::SAMPLES, $node->samples());
        $this->assertEquals(self::LABELS, $node->labels());
        $this->assertEquals(self::CENTER, $node->center());
        $this->assertEquals(self::RADIUS, $node->radius());
    }

    public function test_samples() : void
    {
        $this->assertEquals(self::SAMPLES, $this->node->samples());
    }

    public function test_labels() : void
    {
        $this->assertEquals(self::LABELS, $this->node->labels());
    }

    public function test_center() : void
    {
        $this->assertEquals(self::CENTER, $this->node->center());
    }

    public function test_radius() : void
    {
        $this->assertEquals(self::RADIUS, $this->node->radius());
    }
}
