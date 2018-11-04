<?php

namespace Rubix\ML\Tests\Manifold;

use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Manifold\TSNE;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

class TSNETest extends TestCase
{
    const TRAIN_SIZE = 30;

    protected $estimator;

    protected $generator;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 0, 0], 3.),
            'green' => new Blob([0, 128, 0], 1.),
            'blue' => new Blob([0, 0, 255], 2.),
        ]);

        $this->estimator = new TSNE(1, 10, 12., 500, 1., 0.2, 1e-6, new Euclidean(), 1e-5, 100);
    }

    public function test_build_embedder()
    {
        $this->assertInstanceOf(TSNE::class, $this->estimator);
        $this->assertInstanceOf(Verbose::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_embed()
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE);

        $embedding = $this->estimator->predict($dataset);

        $this->assertCount(self::TRAIN_SIZE, $embedding);

        // file_put_contents('test.json', json_encode($embedding, JSON_PRETTY_PRINT));
    }
}
