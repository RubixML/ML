<?php

namespace Rubix\ML\Tests\Embedders;

use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Embedders\TSNE;
use Rubix\ML\Other\Loggers\BlackHole;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\TestCase;

/**
 * @group Embedders
 * @covers \Rubix\ML\Embedders\TSNE
 */
class TSNETest extends TestCase
{
    /**
     * The number of samples in the validation set.
     *
     * @var int
     */
    protected const TEST_SIZE = 30;

    /**
     * Constant used to see the random number generator.
     *
     * @var int
     */
    protected const RANDOM_SEED = 0;

    /**
     * @var \Rubix\ML\Datasets\Generators\Agglomerate
     */
    protected $generator;

    /**
     * @var \Rubix\ML\Embedders\TSNE
     */
    protected $embedder;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 32, 0], 30.0),
            'green' => new Blob([0, 128, 0], 10.0),
            'blue' => new Blob([0, 32, 255], 20.0),
        ], [2, 3, 4]);

        $this->embedder = new TSNE(1, 10.0, 10, 12.0, 500, 1e-7, 10, new Euclidean());

        $this->embedder->setLogger(new BlackHole());

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(TSNE::class, $this->embedder);
        $this->assertInstanceOf(Verbose::class, $this->embedder);
    }

    /**
     * @test
     */
    public function badNumDimensions() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(0);
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->embedder->compatibility());
    }

    /**
     * @test
     */
    public function params() : void
    {
        $expected = [
            'dimensions' => 1,
            'rate' => 10.0,
            'perplexity' => 10,
            'exaggeration' => 12.0,
            'epochs' => 500,
            'min_gradient' => 1.0E-7,
            'window' => 10,
            'kernel' => new Euclidean(),
        ];

        $this->assertEquals($expected, $this->embedder->params());
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $dataset = $this->generator->generate(self::TEST_SIZE);

        $dataset->apply($this->embedder);

        $this->assertCount(self::TEST_SIZE, $dataset);
        $this->assertCount(1, $dataset->sample(0));

        $steps = $this->embedder->steps();

        $this->assertIsArray($steps);
        $this->assertContainsOnly('float', $steps);
    }
}
