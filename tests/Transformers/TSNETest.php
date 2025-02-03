<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Transformers\TSNE;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(TSNE::class)]
class TSNETest extends TestCase
{
    /**
     * The number of samples in the validation set.
     */
    protected const int TEST_SIZE = 30;

    /**
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected Agglomerate $generator;

    protected TSNE $embedder;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'red' => new Blob([255, 32, 0], 30.0),
                'green' => new Blob([0, 128, 0], 10.0),
                'blue' => new Blob([0, 32, 255], 20.0),
            ],
            weights: [2, 3, 4]
        );

        $this->embedder = new TSNE(
            dimensions: 1,
            rate: 10.0,
            perplexity: 10,
            exaggeration: 12.0,
            epochs: 500,
            minGradient: 1e-7,
            window: 10,
            kernel: new Euclidean()
        );

        $this->embedder->setLogger(new BlackHole());

        srand(self::RANDOM_SEED);
    }

    public function testBadNumDimensions() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(dimensions: 0);
    }

    public function testCompatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->embedder->compatibility());
    }

    public function testTransform() : void
    {
        $dataset = $this->generator->generate(self::TEST_SIZE);

        $dataset->apply($this->embedder);

        $this->assertCount(self::TEST_SIZE, $dataset);
        $this->assertCount(1, $dataset->sample(0));

        $losses = $this->embedder->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnlyFloat($losses);
    }
}
