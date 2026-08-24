<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Transformers\TSNE;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Tensor\Matrix;
use ReflectionMethod;
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

    public function testBadRate() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(rate: 0.0);
    }

    public function testBadPerplexity() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(perplexity: 0);
    }

    public function testBadExaggeration() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(exaggeration: 0.5);
    }

    public function testBadEpochs() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(epochs: 0);
    }

    public function testBadMinGradient() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(minGradient: -1.0);
    }

    public function testDefaults() : void
    {
        $tsne = new TSNE();

        $this->assertEquals(
            't-SNE (dimensions: 2, rate: 100, perplexity: 30, exaggeration: 12, epochs: 1000, '
            . 'min gradient: 1.0E-7, kernel: Euclidean)',
            (string) $tsne
        );
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

    public function testTransformDefaultDimensions() : void
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

    public function testEarlyStopOnMinGradient() : void
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

    public function testRunsAllEpochs() : void
    {
        srand(self::RANDOM_SEED);

        $this->generator->generate(self::TEST_SIZE)->apply($this->embedder);

        $this->assertCount(500, $this->embedder->losses());
    }

    public function testSteps() : void
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

    public function testKernelAffectsPairwiseDistances() : void
    {
        $samples = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 1.0, 0.0],
        ];

        $euclidean = $this->invokePairwiseDistances($this->makeEmbedder(new Euclidean()), $samples);

        $manhattan = $this->invokePairwiseDistances($this->makeEmbedder(new Manhattan()), $samples);

        $this->assertNotEquals($euclidean, $manhattan);

        foreach ($euclidean as $i => $row) {
            $this->assertCount(3, $row);
            $this->assertSame(0.0, $row[$i]);
        }
    }

    public function testAffinitiesNormalize() : void
    {
        $distances = [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ];

        $affinities = $this->invokeAffinities($this->embedder, $distances);

        $this->assertCount(4, $affinities);

        foreach ($affinities as $i => $row) {
            $this->assertCount(4, $row);
            $this->assertEqualsWithDelta(1.0, array_sum($row), 1e-8);
            $this->assertSame(0.0, $row[$i]);
        }
    }

    public function testGradientShape() : void
    {
        $p = Matrix::quick([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.2, 0.3, 0.5],
        ]);

        $y = Matrix::quick([
            [0.0, 1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
        ]);

        $distances = Matrix::quick([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]);

        $rows = $this->invokeGradient($this->embedder, $p, $y, $distances)->asArray();

        $this->assertCount(3, $rows);

        foreach ($rows as $row) {
            $this->assertCount(2, $row);
        }
    }

    public function testGradientNonZero() : void
    {
        $p = Matrix::quick([
            [0.9, 0.05, 0.05],
            [0.5, 0.4, 0.1],
            [0.1, 0.2, 0.7],
        ]);

        $y = Matrix::quick([
            [0.0, 1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
        ]);

        $distances = Matrix::quick([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]);

        $this->assertGreaterThan(
            0.0,
            $this->invokeGradient($this->embedder, $p, $y, $distances)->l2Norm()
        );
    }

    public function testAttenuate() : void
    {
        $this->assertEqualsWithDelta(1.2, $this->invokeAttenuate(1.0, -1.0), 1e-12);
        $this->assertEqualsWithDelta(0.8, $this->invokeAttenuate(1.0, 1.0), 1e-12);
        $this->assertEqualsWithDelta(0.01, $this->invokeAttenuate(0.005, 1.0), 1e-12);
        $this->assertEqualsWithDelta(0.205, $this->invokeAttenuate(0.005, -1.0), 1e-12);
    }

    /**
     * @param Euclidean|Manhattan $kernel
     * @return TSNE
     */
    private function makeEmbedder(Euclidean|Manhattan $kernel) : TSNE
    {
        $embedder = new TSNE(
            dimensions: 2,
            rate: 10.0,
            perplexity: 10,
            exaggeration: 12.0,
            epochs: 20,
            minGradient: 1e-7,
            kernel: $kernel
        );

        $embedder->setLogger(new BlackHole());

        return $embedder;
    }

    /**
     * @param TSNE $embedder
     * @param Matrix $p
     * @param Matrix $y
     * @param Matrix $distances
     * @return Matrix
     */
    private function invokeGradient(TSNE $embedder, Matrix $p, Matrix $y, Matrix $distances) : Matrix
    {
        $method = new ReflectionMethod(TSNE::class, 'gradient');

        $method->setAccessible(true);

        return $method->invokeArgs($embedder, [$p, $y, $distances]);
    }

    /**
     * @param TSNE $embedder
     * @param array<float[]> $distances
     * @return array<float[]>
     */
    private function invokeAffinities(TSNE $embedder, array $distances) : array
    {
        $method = new ReflectionMethod(TSNE::class, 'affinities');

        $method->setAccessible(true);

        return $method->invokeArgs($embedder, [$distances]);
    }

    /**
     * @param float $gain
     * @param float $direction
     * @return float
     */
    private function invokeAttenuate(float $gain, float $direction) : float
    {
        $method = new ReflectionMethod(TSNE::class, 'attenuate');

        $method->setAccessible(true);

        return (float) $method->invokeArgs($this->embedder, [$gain, $direction]);
    }

    /**
     * @param TSNE $embedder
     * @param array<mixed[]> $samples
     * @return array<float[]>
     */
    private function invokePairwiseDistances(TSNE $embedder, array $samples) : array
    {
        $method = new ReflectionMethod(TSNE::class, 'pairwiseDistances');

        $method->setAccessible(true);

        return $method->invokeArgs($embedder, [$samples]);
    }
}
