<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Clusterers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Clusterers\EMeans;
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Cosine;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Clusterers\Seeders\Random;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\MissingExtension;
use Rubix\ML\Exceptions\RuntimeException;
use Tensor\Matrix;
use PHPUnit\Framework\TestCase;

use function array_sum;
use function count;
use function min;
use function array_fill;
use function extension_loaded;

#[Group('Clusterers')]
#[CoversClass(EMeans::class)]
class EMeansTest extends TestCase
{
    /**
     * The number of samples in the training set.
     */
    protected const int TRAIN_SIZE = 512;

    /**
     * The number of samples in the validation set.
     */
    protected const int TEST_SIZE = 256;

    /**
     * The minimum validation score required to pass the test.
     */
    protected const float MIN_SCORE = 0.9;

    /**
     * Constant used to seed the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected Agglomerate $generator;

    protected EMeans $estimator;

    protected VMeasure $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'red' => new Blob(center: [255, 32, 0], stdDev: 50.0),
                'green' => new Blob(center: [0, 128, 0], stdDev: 10.0),
                'blue' => new Blob(center: [0, 32, 255], stdDev: 30.0),
            ],
            weights: [0.5, 0.2, 0.3]
        );

        $this->estimator = new EMeans(
            k: 3,
            epochs: 5.0,
            kernel: new Euclidean(),
            seeder: new Random()
        );

        $this->metric = new VMeasure();

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function preConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    #[Test]
    public function type() : void
    {
        $this->assertEquals(EstimatorType::clusterer(), $this->estimator->type());
    }

    #[Test]
    public function compatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    public function params() : void
    {
        $expected = [
            'k' => 3,
            'epochs' => 5.0,
            'batch size' => null,
            'initial step' => null,
            'kernel' => new Euclidean(),
            'seeder' => new Random(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    public function badK() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new EMeans(k: 0);
    }

    #[Test]
    public function badEpochs() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new EMeans(k: 3, epochs: 0.0);
    }

    #[Test]
    public function negativeEpochs() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new EMeans(k: 3, epochs: -1.0);
    }

    #[Test]
    public function nonFiniteEpochs() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new EMeans(k: 3, epochs: NAN);
    }

    #[Test]
    public function badBatchSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new EMeans(k: 3, batchSize: 0);
    }

    #[Test]
    public function negativeBatchSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new EMeans(k: 3, batchSize: -4);
    }

    #[Test]
    public function badAlpha() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new EMeans(k: 3, alpha0: 0.0);
    }

    #[Test]
    public function negativeAlpha() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new EMeans(k: 3, alpha0: -0.5);
    }

    #[Test]
    public function nonFiniteAlpha() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new EMeans(k: 3, alpha0: INF);
    }

    #[Test]
    public function invalidKernel() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new EMeans(k: 3, kernel: new Cosine());
    }

    #[Test]
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick(samples: [['bad']]));
    }

    #[Test]
    public function trainMissingTensorExtension() : void
    {
        if (extension_loaded('tensor')) {
            $this->markTestSkipped('The tensor extension is loaded.');
        }

        $this->expectException(MissingExtension::class);

        $this->estimator->train(Unlabeled::quick(samples: [[1.0, 2.0, 3.0]]));
    }

    #[Test]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick(samples: [[1.0, 2.0, 3.0]]));
    }

    #[Test]
    public function progressContract() : void
    {
        $this->skipIfTensorAbsent();

        $this->assertSame([], iterator_to_array($this->estimator->progress(), false));

        srand(self::RANDOM_SEED);

        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertNotEmpty($losses);
        $this->assertNotContainsNAN(array_values($losses));

        $rows = iterator_to_array($this->estimator->progress(), false);

        $this->assertCount(count($losses), $rows);

        foreach ($rows as $row) {
            $this->assertIsInt($row['Epoch']);
            $this->assertIsFloat($row['Inertia']);
        }

        $this->assertSame(array_values($losses), array_column($rows, 'Inertia'));
        $this->assertSame(array_keys($losses), array_column($rows, 'Epoch'));
    }

    #[Test]
    public function trainPredict() : void
    {
        $this->skipIfTensorAbsent();

        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $centroids = $this->estimator->centroids();

        $this->assertIsArray($centroids);
        $this->assertCount(3, $centroids);
        $this->assertContainsOnlyArray($centroids);
        $this->assertNotContainsNAN($centroids);

        $sizes = $this->estimator->sizes();

        $this->assertIsArray($sizes);
        $this->assertCount(3, $sizes);
        $this->assertContainsOnlyInt($sizes);
        $this->assertSame(self::TRAIN_SIZE, array_sum($sizes));
        $this->assertGreaterThan(0, min($sizes));

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $testing->labels()
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);

        $proba = $this->estimator->proba($testing);

        $this->assertCount(self::TEST_SIZE, $proba);

        foreach ($proba as $row) {
            $this->assertCount(3, $row);
            $this->assertNotContainsNAN($row);
            $this->assertEqualsWithDelta(1.0, array_sum($row), 1e-6);
        }
    }

    #[Test]
    public function hyperparameterResolution() : void
    {
        $this->skipIfTensorAbsent();

        srand(self::RANDOM_SEED);

        $training = $this->generator->generate(256);

        $estimator = new EMeans(k: 3, epochs: 2.5);

        $estimator->setLogger(new BlackHole());

        $estimator->train($training);

        $params = $estimator->params();

        $this->assertIsInt($params['resolved batch size']);
        $this->assertGreaterThan(0, $params['resolved batch size']);
        $this->assertLessThanOrEqual(256, $params['resolved batch size']);

        $batch = $params['resolved batch size'];

        $this->assertSame(0, $batch & ($batch - 1), 'The batch size must be a power of 2.');

        $this->assertIsFloat($params['resolved initial step']);
        $this->assertGreaterThanOrEqual(1.0e-4, $params['resolved initial step']);
        $this->assertLessThanOrEqual(1.0 - 1.0e-10, $params['resolved initial step']);
    }

    #[Test]
    public function explicitAlphaIsClamped() : void
    {
        $this->skipIfTensorAbsent();

        srand(self::RANDOM_SEED);

        $training = $this->generator->generate(64);

        $estimator = new EMeans(k: 2, epochs: 1.0, alpha0: 4.0);

        $estimator->setLogger(new BlackHole());

        $estimator->train($training);

        $this->assertEqualsWithDelta(1.0 - 1.0e-10, $estimator->params()['resolved initial step'], 1e-12);

        $estimator = new EMeans(k: 2, epochs: 1.0, alpha0: 1.0e-6);

        $estimator->setLogger(new BlackHole());

        $estimator->train($training);

        $this->assertEqualsWithDelta(1.0e-6, $estimator->params()['resolved initial step'], 1e-12);
    }

    #[Test]
    public function scheduledAlpha() : void
    {
        $accessor = new EMeansAccessor(k: 3);

        $this->assertSame(0.5, $accessor->scheduledAlpha(0.5, 0, 4));

        $alphas = [];

        for ($i = 0; $i <= 10; ++$i) {
            $alphas[] = $accessor->scheduledAlpha(0.5, $i, 10);
        }

        for ($i = 0; $i < count($alphas) - 1; ++$i) {
            $this->assertGreaterThanOrEqual($alphas[$i + 1] - 1e-15, $alphas[$i]);
        }

        $this->assertEqualsWithDelta(0.05, $alphas[count($alphas) - 1], 1e-12);
    }

    #[Test]
    public function epochRowCounts() : void
    {
        $accessor = new EMeansAccessor(k: 3);

        $this->assertSame([100, 100, 100], $accessor->withEpochs(3.0)->epochRowCounts(100));

        $this->assertSame([100, 50, 100], $accessor->withEpochs(2.5)->epochRowCounts(100));

        $this->assertSame([50, 100], $accessor->withEpochs(1.5)->epochRowCounts(100));

        $this->assertSame([40], $accessor->withEpochs(0.4)->epochRowCounts(100));

        $this->assertSame([1], $accessor->withEpochs(0.01)->epochRowCounts(5));
    }

    #[Test]
    public function rowArgmins() : void
    {
        $accessor = new EMeansAccessor(k: 2);

        $matrix = Matrix::fromArray([
            [3.0, 1.0, 2.0],
            [5.0, 5.0, 1.0],
            [2.0, 2.0, 9.0],
        ], false);

        $this->assertSame([1, 2, 0], $accessor->rowArgmins($matrix));
    }

    #[Test]
    public function tiesBreakToFirstCluster() : void
    {
        $this->skipIfTensorAbsent();

        $estimator = new EMeans(
            k: 2,
            epochs: 1.0,
            seeder: new DuplicateSeeder()
        );

        $estimator->setLogger(new BlackHole());

        $estimator->train(Unlabeled::quick(samples: [[1.0, 2.0]]));

        $this->assertTrue($estimator->trained());

        $this->assertSame(1, array_sum($estimator->sizes()));

        $this->assertSame(0, $estimator->predictSample([1.0, 2.0]));
    }

    #[Test]
    public function restoreStateFromSerializedModel() : void
    {
        $this->skipIfTensorAbsent();

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $restored = unserialize(serialize($this->estimator));

        $this->assertTrue($restored->trained());

        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->assertEquals($this->estimator->predict($testing), $restored->predict($testing));
    }

    #[Test]
    public function quality() : void
    {
        $this->skipIfTensorAbsent();

        srand(self::RANDOM_SEED);

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->setLogger(new BlackHole());

        $this->estimator->train($training);

        $estimatorDistance = $this->meanNearestCentroidDistance(
            $testing->samples(),
            $this->estimator->centroids()
        );

        srand(self::RANDOM_SEED);

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $baseline = new KMeans(
            k: 3,
            epochs: 30,
            minChange: 1e-4
        );

        $baseline->setLogger(new BlackHole());

        $baseline->train($training);

        $baselineDistance = $this->meanNearestCentroidDistance(
            $testing->samples(),
            $baseline->centroids()
        );

        $this->assertLessThanOrEqual($baselineDistance, $estimatorDistance);
    }

    protected function skipIfTensorAbsent() : void
    {
        if (!extension_loaded('tensor')) {
            $this->markTestSkipped('The tensor C extension is required.');
        }
    }

    /**
     * Compute the mean distance from the test samples to their nearest
     * centroid.
     *
     * @param list<list<float>> $samples
     * @param list<list<float>> $centroids
     * @return float
     */
    protected function meanNearestCentroidDistance(array $samples, array $centroids) : float
    {
        $kernel = new Euclidean();

        $sum = 0.0;

        foreach ($samples as $sample) {
            $best = INF;

            foreach ($centroids as $centroid) {
                $distance = $kernel->compute($sample, $centroid);

                if ($distance !== null and $distance < $best) {
                    $best = $distance;
                }
            }

            $sum += $best;
        }

        return $sum / count($samples);
    }

    protected function assertNotContainsNAN(array $values) : void
    {
        foreach ($values as $value) {
            if (is_array($value)) {
                $this->assertNotContainsNAN($value);
            } else {
                $this->assertIsFloat($value);

                $this->assertNotSame(NAN, $value);
            }
        }
    }
}

/**
 * A test-local seeder that seeds every cluster with the very first sample
 * of the dataset, forcing exact ties between the cluster positions.
 */
class DuplicateSeeder implements Seeder
{
    /**
     * Seed k cluster centroids with the same sample.
     *
     * @param Dataset $dataset
     * @param int $k
     * @return list<list<string|int|float>>
     */
    public function seed(Dataset $dataset, int $k) : array
    {
        return array_fill(0, $k, $dataset->sample(0));
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Duplicate';
    }
}

/**
 * A test-local subclass that exposes the protected scheduling helpers of
 * the E Means estimator for white-box testing.
 */
class EMeansAccessor extends EMeans
{
    public function withEpochs(float $epochs) : self
    {
        return new self(
            k: $this->k,
            epochs: $epochs
        );
    }

    public function scheduledAlpha(float $alpha0, int $step, int $total) : float
    {
        return parent::scheduledAlpha($alpha0, $step, $total);
    }

    public function epochRowCounts(int $n) : array
    {
        return parent::epochRowCounts($n);
    }

    public function rowArgmins(Matrix $matrix) : array
    {
        return parent::rowArgmins($matrix);
    }
}
