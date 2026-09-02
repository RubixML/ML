<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use ReflectionMethod;
use NDArray;
use NumPower;
use Rubix\ML\DataType;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Transformers\TSNE;
use Rubix\ML\Datasets\Generators\Blob;
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
            minGradient: 0.0
        );

        $this->embedder->setLogger(new BlackHole());

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function badNumDimensions() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(dimensions: 0);
    }

    #[Test]
    public function badRate() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(rate: 0.0);
    }

    #[Test]
    public function badPerplexity() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(perplexity: 0);
    }

    #[Test]
    public function badExaggeration() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(exaggeration: 0.5);
    }

    #[Test]
    public function badEpochs() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(epochs: 0);
    }

    #[Test]
    public function badMinGradient() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(minGradient: -1.0);
    }

    #[Test]
    public function defaults() : void
    {
        $tsne = new TSNE();

        $this->assertEquals(
            't-SNE (dimensions: 2, rate: 100, perplexity: 30, exaggeration: 12, epochs: 1000, '
            . 'min gradient: 1.0E-7)',
            (string) $tsne
        );
    }

    #[Test]
    public function compatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->embedder->compatibility());
    }

    #[Test]
    public function transform() : void
    {
        $dataset = $this->generator->generate(self::TEST_SIZE);

        $dataset->apply($this->embedder);

        $this->assertCount(self::TEST_SIZE, $dataset);
        $this->assertCount(1, $dataset->sample(0));

        $losses = $this->embedder->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnlyFloat($losses);
    }

    #[Test]
    public function transformDefaultDimensions() : void
    {
        srand(self::RANDOM_SEED);

        $dataset = $this->generator->generate(self::TEST_SIZE);

        $embedder = new TSNE();

        $embedder->setLogger(new BlackHole());

        $dataset->apply($embedder);

        $this->assertCount(self::TEST_SIZE, $dataset);
        $this->assertCount(2, $dataset->sample(0));

        $this->assertIsArray($embedder->losses());
    }

    #[Test]
    public function earlyStopOnMinGradient() : void
    {
        srand(self::RANDOM_SEED);

        $earlyStop = new TSNE(
            dimensions: 1,
            epochs: 200,
            minGradient: 1e9
        );

        $earlyStop->setLogger(new BlackHole());

        $this->generator->generate(self::TEST_SIZE)->apply($earlyStop);

        $this->assertCount(1, $earlyStop->losses());

        $steps = iterator_to_array($earlyStop->steps());

        $this->assertCount(1, $steps);
    }

    #[Test]
    public function runsAllEpochs() : void
    {
        srand(self::RANDOM_SEED);

        $this->generator->generate(self::TEST_SIZE)->apply($this->embedder);

        $this->assertCount(500, $this->embedder->losses());
    }

    #[Test]
    public function steps() : void
    {
        srand(self::RANDOM_SEED);

        $this->generator->generate(self::TEST_SIZE)->apply($this->embedder);

        $steps = iterator_to_array($this->embedder->steps());

        $this->assertCount(500, $steps);

        $epoch = 0;

        $losses = [];

        foreach ($steps as $step) {
            $this->assertSame($epoch, $step['epoch']);
            $this->assertIsFloat($step['loss']);
            $this->assertGreaterThanOrEqual(0.0, $step['loss']);
            $losses[] = $step['loss'];
            ++$epoch;
        }

        $this->assertEquals($losses, $this->embedder->losses());
    }

    #[Test]
    public function gradientShape() : void
    {
        $p = NumPower::array([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.2, 0.3, 0.5],
        ], 'float32');

        $y = NumPower::array([
            [0.0, 1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
        ], 'float32');

        $distances = NumPower::array([
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 1.0],
            [4.0, 1.0, 0.0],
        ], 'float32');

        $rows = $this->invokeGradient($this->embedder, $p, $y, $distances)->toArray();

        $this->assertCount(3, $rows);

        foreach ($rows as $row) {
            $this->assertCount(2, $row);
        }
    }

    #[Test]
    public function gradientNonZero() : void
    {
        $p = NumPower::array([
            [0.9, 0.05, 0.05],
            [0.5, 0.4, 0.1],
            [0.1, 0.2, 0.7],
        ], 'float32');

        $y = NumPower::array([
            [0.0, 1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
        ], 'float32');

        $distances = NumPower::array([
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 1.0],
            [4.0, 1.0, 0.0],
        ], 'float32');

        $gradient = $this->invokeGradient($this->embedder, $p, $y, $distances);

        $this->assertGreaterThan(
            0.0,
            NumPower::sqrt(NumPower::sum(NumPower::square($gradient)))
        );
    }

    /**
     * @param TSNE $embedder
     * @param NDArray $p
     * @param NDArray $y
     * @param NDArray $distances
     * @return NDArray
     */
    private function invokeGradient(TSNE $embedder, NDArray $p, NDArray $y, NDArray $distances) : NDArray
    {
        $method = new ReflectionMethod(TSNE::class, 'gradient');

        $method->setAccessible(true);

        return $method->invokeArgs($embedder, [$p, $y, $distances]);
    }
}
