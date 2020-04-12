<?php

namespace Rubix\ML\Tests\Graph\Trees;

use Rubix\ML\Graph\Trees\Tree;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Graph\Trees\BinaryTree;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

/**
 * @group Trees
 * @covers \Rubix\ML\Graph\Trees\KDTree
 */
class KDTreeTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    /**
     * @var \Rubix\ML\Datasets\Generators\Agglomerate
     */
    protected $generator;

    /**
     * @var \Rubix\ML\Graph\Trees\KDTree
     */
    protected $tree;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            'east' => new Blob([5, -2, -2]),
            'west' => new Blob([0, 5, -3]),
        ], [0.5, 0.5]);

        $this->tree = new KDTree(20, new Euclidean());

        srand(self::RANDOM_SEED);
    }

    protected function assertPreConditions() : void
    {
        $this->assertEquals(0, $this->tree->height());
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(KDTree::class, $this->tree);
        $this->assertInstanceOf(Spatial::class, $this->tree);
        $this->assertInstanceOf(BinaryTree::class, $this->tree);
        $this->assertInstanceOf(Tree::class, $this->tree);
    }

    /**
     * @test
     */
    public function growNeighborsRange() : void
    {
        $this->tree->grow($this->generator->generate(50));

        $this->assertGreaterThan(2, $this->tree->height());

        $sample = $this->generator->generate(1)->sample(0);

        [$samples, $labels, $distances] = $this->tree->nearest($sample, 5);

        $this->assertCount(5, $samples);
        $this->assertCount(5, $labels);
        $this->assertCount(5, $distances);

        $this->assertCount(1, array_unique($labels));

        [$samples, $labels, $distances] = $this->tree->range($sample, 5.);

        $this->assertCount(25, $samples);
        $this->assertCount(25, $labels);
        $this->assertCount(25, $distances);

        $this->assertCount(1, array_unique($labels));
    }
}
