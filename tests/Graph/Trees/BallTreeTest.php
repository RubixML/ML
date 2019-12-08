<?php

namespace Rubix\ML\Tests\Graph\Trees;

use Rubix\ML\Graph\Trees\Tree;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Graph\Trees\BinaryTree;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

class BallTreeTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    /**
     * @var \Rubix\ML\Datasets\Generators\Generator
     */
    protected $generator;

    /**
     * @var \Rubix\ML\Graph\Trees\BallTree
     */
    protected $tree;

    public function setUp() : void
    {
        $this->generator = new Agglomerate([
            'east' => new Blob([5, -2, -2]),
            'west' => new Blob([0, 5, -3]),
        ], [0.5, 0.5]);

        $this->tree = new BallTree(20, new Euclidean());

        srand(self::RANDOM_SEED);
    }

    public function test_build_tree() : void
    {
        $this->assertInstanceOf(BallTree::class, $this->tree);
        $this->assertInstanceOf(Spatial::class, $this->tree);
        $this->assertInstanceOf(BinaryTree::class, $this->tree);
        $this->assertInstanceOf(Tree::class, $this->tree);

        $this->assertEquals(0, $this->tree->height());
    }

    public function test_grow_neighbors_range() : void
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
