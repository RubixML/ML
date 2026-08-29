<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Graph\Trees;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Kernels\Distance\Canberra;
use Rubix\ML\Kernels\Distance\Cosine;
use Rubix\ML\Kernels\Distance\Diagonal;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Gower;
use Rubix\ML\Kernels\Distance\Hamming;
use Rubix\ML\Kernels\Distance\Jaccard;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\ML\Kernels\Distance\SafeEuclidean;
use Rubix\ML\Kernels\Distance\SparseCosine;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

#[Group('Trees')]
#[CoversClass(BallTree::class)]
class BallTreeTest extends TestCase
{
    protected const int DATASET_SIZE = 100;

    protected const int RANDOM_SEED = 0;

    protected Agglomerate $generator;

    protected BallTree $tree;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'east' => new Blob(center: [5, -2, -2]),
                'west' => new Blob(center: [0, 5, -3]),
            ],
            weights: [0.5, 0.5]
        );

        $this->tree = new BallTree(
            20,
            new Euclidean()
        );

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function preConditions() : void
    {
        $this->assertEquals(0, $this->tree->height());
    }

    #[Test]
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

        [$samples, $labels, $distances] = $this->tree->range($sample, 4.3);

        $this->assertCount(50, $samples);
        $this->assertCount(50, $labels);
        $this->assertCount(50, $distances);

        $this->assertCount(1, array_unique($labels));
    }

    #[Test]
    public function growWithSameSamples() : void
    {
        $generator = new Agglomerate(generators: [
            'east' => new Blob(center: [5, -2, 10], stdDev: 0.0),
        ]);

        $dataset = $generator->generate(self::DATASET_SIZE);

        $this->tree->grow($dataset);

        $this->assertEquals(2, $this->tree->height());
    }

    #[Test]
    public function rejectCosineKernel() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new BallTree(kernel: new Cosine());
    }

    #[Test]
    public function rejectSparseCosineKernel() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new BallTree(kernel: new SparseCosine());
    }

    #[Test]
    public function compatibleKernels() : void
    {
        $kernels = [
            new Euclidean(),
            new Manhattan(),
            new Minkowski(),
            new SafeEuclidean(),
            new Diagonal(),
            new Canberra(),
            new Gower(),
            new Hamming(),
            new Jaccard(),
        ];

        foreach ($kernels as $kernel) {
            new BallTree(kernel: $kernel);
        }

        $this->assertTrue(true);
    }
}
