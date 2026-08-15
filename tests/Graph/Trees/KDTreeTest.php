<?php

namespace Rubix\ML\Tests\Graph\Trees;

use Rubix\ML\Graph\Trees\Tree;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Graph\Trees\BinaryTree;
use Rubix\ML\Datasets\Labeled;
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
    protected const DATASET_SIZE = 100;

    protected const RANDOM_SEED = 0;

    /**
     * @var Agglomerate
     */
    protected $generator;

    /**
     * @var KDTree
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
        $this->tree->grow($this->generator->generate(self::DATASET_SIZE));

        $this->assertGreaterThan(2, $this->tree->height());

        $sample = $this->generator->generate(1)->sample(0);

        [$samples, $labels, $distances] = $this->tree->nearest($sample, 5);

        $this->assertCount(5, $samples);
        $this->assertCount(5, $labels);
        $this->assertCount(5, $distances);

        $this->assertCount(1, array_unique($labels));

        [$samples, $labels, $distances] = $this->tree->range($sample, 5.0);

        $this->assertCount(51, $samples);
        $this->assertCount(51, $labels);
        $this->assertCount(51, $distances);

        $this->assertCount(2, array_unique($labels));
    }

    /**
     * @test
     */
    public function growWithSameSamples() : void
    {
        $generator = new Agglomerate([
            'east' => new Blob([5, -2, 10], 0.0),
        ]);

        $dataset = $generator->generate(self::DATASET_SIZE);

        $this->tree->grow($dataset);

        $this->assertEquals(2, $this->tree->height());
    }

    /**
     * @test
     */
    public function nearestMatchesBruteForce() : void
    {
        $samples = [
            [7.2, 5.7],
            [8.7, 6.7],
            [9.2, 5.9],
            [1.8, 7.9],
            [0.2, 5.4],
            [0.6, 9.3],
            [0.6, 0.8],
            [2.1, 7.1],
            [4.3, 4.5],
        ];

        $labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'];

        $tree = new KDTree(5, new Euclidean());

        $tree->grow(Labeled::quick($samples, $labels));

        [$neighbors, $neighborLabels, $distances] = $tree->nearest([6.8, 1.3], 1);

        $this->assertEquals([[4.3, 4.5]], $neighbors);
        $this->assertEquals(['i'], $neighborLabels);
        $this->assertEqualsWithDelta(4.06078810084939, $distances[0], 1e-6);
    }

    /**
     * @test
     */
    public function rangeMatchesBruteForce() : void
    {
        $samples = [];

        for ($x = 0; $x <= 10; $x += 2) {
            for ($y = 0; $y <= 10; $y += 2) {
                $samples[] = [(float) $x, (float) $y];
            }
        }

        $labels = array_map('strval', range(0, 35));

        $tree = new KDTree(5, new Euclidean());

        $tree->grow(Labeled::quick($samples, $labels));

        [$neighbors, $neighborLabels, $distances] = $tree->range([5.0, 15.0], 5.1);

        $this->assertCount(2, $neighbors);
        $this->assertCount(2, $neighborLabels);
        $this->assertCount(2, $distances);

        $this->assertEqualsWithDelta(5.09901951359278, $distances[0], 1e-6);
        $this->assertEqualsWithDelta(5.09901951359278, $distances[1], 1e-6);
    }
}
