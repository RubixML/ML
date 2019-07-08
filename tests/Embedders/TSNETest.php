<?php

namespace Rubix\ML\Tests\Embedders;

use Rubix\ML\Verbose;
use Rubix\ML\Embedders\TSNE;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Loggers\BlackHole;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class TSNETest extends TestCase
{
    protected const TRAIN_SIZE = 30;

    protected const RANDOM_SEED = 0;

    protected $embedder;

    protected $generator;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 32, 0], 30.),
            'green' => new Blob([0, 128, 0], 10.),
            'blue' => new Blob([0, 32, 255], 20.),
        ], [2, 3, 4]);

        $this->embedder = new TSNE(1, 10, 12., 10., new Euclidean(), 500, 1e-7, 5);

        $this->embedder->setLogger(new BlackHole());

        srand(self::RANDOM_SEED);
    }

    public function test_build_embedder()
    {
        $this->assertInstanceOf(TSNE::class, $this->embedder);
        $this->assertInstanceOf(Verbose::class, $this->embedder);

        $this->assertNotContains(DataType::CATEGORICAL, $this->embedder->compatibility());
        $this->assertContains(DataType::CONTINUOUS, $this->embedder->compatibility());
    }

    public function test_embed()
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE);

        $samples = $this->embedder->embed($dataset);

        $this->assertCount(self::TRAIN_SIZE, $samples);
        $this->assertCount(1, $samples[0]);

        // file_put_contents('embedding.json', json_encode($samples, JSON_PRETTY_PRINT));
    }

    public function test_embed_incompatible()
    {
        $this->expectException(InvalidArgumentException::class);

        $this->embedder->embed(Unlabeled::quick([['bad']]));
    }
}
