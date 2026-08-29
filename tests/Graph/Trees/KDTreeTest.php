<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Graph\Trees;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Kernels\Distance\Canberra;
use Rubix\ML\Kernels\Distance\Cosine;
use Rubix\ML\Kernels\Distance\Diagonal;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Gower;
use Rubix\ML\Kernels\Distance\Jaccard;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\ML\Kernels\Distance\SafeEuclidean;
use Rubix\ML\Kernels\Distance\SparseCosine;
use PHPUnit\Framework\TestCase;

#[Group('Trees')]
#[CoversClass(KDTree::class)]
class KDTreeTest extends TestCase
{
    protected const int DATASET_SIZE = 100;

    protected const int RANDOM_SEED = 0;

    protected Agglomerate $generator;

    protected KDTree $tree;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'east' => new Blob(center: [5, -2, -2]),
                'west' => new Blob(center: [0, 5, -3]),
            ],
            weights: [0.5, 0.5]
        );

        $this->tree = new KDTree(
            maxLeafSize: 20,
            kernel: new Euclidean()
        );

        srand(self::RANDOM_SEED);
    }

    public function testAssertPreConditions() : void
    {
        $this->assertEquals(0, $this->tree->height());
    }

    public function testGrowNeighborsRange() : void
    {
        $this->tree->grow($this->generator->generate(self::DATASET_SIZE));

        $this->assertGreaterThan(2, $this->tree->height());

        $sample = $this->generator->generate(1)->sample(0);

        [$samples, $labels, $distances] = $this->tree->nearest(sample: $sample, k: 5);

        $this->assertCount(5, $samples);
        $this->assertCount(5, $labels);
        $this->assertCount(5, $distances);

        $this->assertCount(1, array_unique($labels));

        [$samples, $labels, $distances] = $this->tree->range(sample: $sample, radius: 5.0);

        $this->assertCount(51, $samples);
        $this->assertCount(51, $labels);
        $this->assertCount(51, $distances);

        $this->assertCount(2, array_unique($labels));
    }

    public function testGrowWithSameSamples() : void
    {
        $generator = new Agglomerate(generators: [
            'east' => new Blob(center: [5, -2, 10], stdDev: 0.0),
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

    public function testRejectCosineKernel() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new KDTree(kernel: new Cosine());
    }

    public function testRejectSparseCosineKernel() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new KDTree(kernel: new SparseCosine());
    }

    public function testRejectJaccardKernel() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new KDTree(kernel: new Jaccard());
    }

    public function testCompatibleKernels() : void
    {
        $kernels = [
            new Euclidean(),
            new Manhattan(),
            new Minkowski(),
            new SafeEuclidean(),
            new Diagonal(),
            new Canberra(),
            new Gower(),
        ];

        foreach ($kernels as $kernel) {
            new KDTree(kernel: $kernel);
        }

        $this->assertTrue(true);
    }
}
