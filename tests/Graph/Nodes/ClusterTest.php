<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Leaf;
use Rubix\ML\Graph\Nodes\Cluster;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Kernels\Distance\Euclidean;
use PHPUnit\Framework\TestCase;

class ClusterTest extends TestCase
{
    protected const SAMPLES = [
        [5., 2., -3],
        [6., 4., -5],
    ];

    protected const LABELS = [
        22, 13,
    ];

    protected const CENTER = [5.5, 3., -4];
    protected const RADIUS = 1.5;

    public function test_build_node()
    {
        $node = new Cluster(self::SAMPLES, self::LABELS, self::CENTER, self::RADIUS);

        $this->assertInstanceOf(Cluster::class, $node);
        $this->assertInstanceOf(BinaryNode::class, $node);
        $this->assertInstanceOf(Leaf::class, $node);
        $this->assertInstanceOf(Node::class, $node);

        $this->assertEquals(self::SAMPLES, $node->samples());
        $this->assertEquals(self::LABELS, $node->labels());
        $this->assertEquals(self::CENTER, $node->center());
        $this->assertEquals(self::RADIUS, $node->radius());
    }

    public function test_terminate()
    {
        $dataset = Labeled::quick(self::SAMPLES, self::LABELS);

        $node = Cluster::terminate($dataset, new Euclidean());

        $this->assertEquals(self::SAMPLES, $node->samples());
        $this->assertEquals(self::LABELS, $node->labels());
        $this->assertEquals(self::CENTER, $node->center());
        $this->assertEquals(self::RADIUS, $node->radius());
    }
}
