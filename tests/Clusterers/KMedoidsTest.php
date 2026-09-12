<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Clusterers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Clusterers\KMedoids;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Hamming;
use Rubix\ML\Clusterers\Seeders\KMC2;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

use function min;
use function in_array;

#[Group('Clusterers')]
#[CoversClass(KMedoids::class)]
class KMedoidsTest extends TestCase
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

    protected KMedoids $estimator;

    protected VMeasure $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'red' => new Blob(
                    center: [255, 32, 0],
                    stdDev: 50.0
                ),
                'green' => new Blob(
                    center: [0, 128, 0],
                    stdDev: 10.0
                ),
                'blue' => new Blob(
                    center: [0, 32, 255],
                    stdDev: 30.0
                ),
            ],
            weights: [0.5, 0.2, 0.3]
        );

        $this->estimator = new KMedoids(
            k: 3,
            batchSize: 50,
            epochs: 10,
            minChange: 1e-4,
            kernel: new Euclidean(),
            seeder: new KMC2()
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
    public function badK() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new KMedoids(k: 0);
    }

    #[Test]
    public function badSampleSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new KMedoids(k: 3, batchSize: 2);
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
    public function compatibilityKernelDependent() : void
    {
        $estimator = new KMedoids(k: 3, kernel: new Hamming());

        $expected = [
            DataType::categorical(),
        ];

        $this->assertEquals($expected, $estimator->compatibility());
    }

    #[Test]
    public function params() : void
    {
        $expected = [
            'k' => 3,
            'batch size' => 50,
            'epochs' => 10,
            'min change' => 1e-4,
            'kernel' => new Euclidean(),
            'seeder' => new KMC2(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    public function trainPredict() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $medoids = $this->estimator->medoids();

        $this->assertIsArray($medoids);
        $this->assertCount(3, $medoids);
        $this->assertContainsOnlyArray($medoids);

        $samples = $training->samples();

        foreach ($medoids as $index => $medoid) {
            $this->assertTrue(
                in_array($medoid, $samples, true),
                "Medoid at offset $index is not a sample in the training set."
            );
        }

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnlyFloat($losses);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $testing->labels()
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function trainCategorical() : void
    {
        srand(self::RANDOM_SEED);

        $rows = [
            ['a', 'x', 'one'],
            ['a', 'x', 'two'],
            ['a', 'x', 'one'],
            ['a', 'x', 'two'],
            ['a', 'x', 'one'],
            ['b', 'y', 'three'],
            ['b', 'y', 'four'],
            ['b', 'y', 'three'],
            ['b', 'y', 'four'],
            ['b', 'y', 'three'],
            ['c', 'z', 'five'],
            ['c', 'z', 'six'],
            ['c', 'z', 'five'],
            ['c', 'z', 'six'],
            ['c', 'z', 'five'],
        ];

        $labels = [
            'red', 'red', 'red', 'red', 'red',
            'green', 'green', 'green', 'green', 'green',
            'blue', 'blue', 'blue', 'blue', 'blue',
        ];

        $training = Labeled::quick($rows, $labels);

        $estimator = new KMedoids(k: 3, batchSize: 15, epochs: 10, kernel: new Hamming());

        $estimator->train($training);

        $this->assertTrue($estimator->trained());

        $samples = $training->samples();

        foreach ($estimator->medoids() as $medoid) {
            $this->assertTrue(in_array($medoid, $samples, true));
        }

        $predictions = $estimator->predict($training);

        $score = $this->metric->score($predictions, $training->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick(samples: [['bad']]));
    }

    #[Test]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick(samples: [[1.0]]));
    }

    #[Test]
    public function restoreStateFromSerializedModel() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $restored = unserialize(serialize($this->estimator));

        $this->assertTrue($restored->trained());

        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->assertEquals($this->estimator->predict($testing), $restored->predict($testing));
    }

    #[Test]
    public function lossesReflectFullDatasetInertia() : void
    {
        srand(self::RANDOM_SEED);

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertCount($this->estimator->params()['epochs'], $losses);

        $medoids = $this->estimator->medoids();

        $kernel = $this->estimator->params()['kernel'];

        $expected = 0.0;

        foreach ($training->samples() as $sample) {
            $min = INF;

            foreach ($medoids as $medoid) {
                $distance = $kernel->compute($sample, $medoid);

                if ($distance < $min) {
                    $min = $distance;
                }
            }

            $expected += $min;
        }

        $expected /= $training->numSamples();

        $winner = null;

        foreach ($losses as $loss) {
            if ($winner === null or $loss < $winner) {
                $winner = $loss;
            }
        }

        $this->assertNotNull($winner);
        $this->assertEqualsWithDelta($expected, $winner, 1e-8);
    }

    #[Test]
    public function claraBestOfRBeatsSingleIteration() : void
    {
        srand(self::RANDOM_SEED);
        $training = $this->generator->generate(self::TRAIN_SIZE);

        srand(self::RANDOM_SEED);
        $single = new KMedoids(k: 3, batchSize: 100, epochs: 1, minChange: 1e-4);
        $single->train($training);

        srand(self::RANDOM_SEED);
        $many = new KMedoids(k: 3, batchSize: 100, epochs: 10, minChange: 1e-4);
        $many->train($training);

        $singleLoss = min($single->losses());
        $manyLoss = min($many->losses());

        $this->assertLessThanOrEqual($singleLoss, $manyLoss);
    }

    #[Test]
    public function seededReproducibility() : void
    {
        srand(self::RANDOM_SEED);
        $training = $this->generator->generate(self::TRAIN_SIZE);
        srand(self::RANDOM_SEED);
        $trainingAgain = $this->generator->generate(self::TRAIN_SIZE);

        srand(self::RANDOM_SEED);
        $estimatorA = new KMedoids(k: 3, batchSize: 100, epochs: 50, kernel: new Euclidean());
        $estimatorA->train($training);

        srand(self::RANDOM_SEED);
        $estimatorB = new KMedoids(k: 3, batchSize: 100, epochs: 50, kernel: new Euclidean());
        $estimatorB->train($trainingAgain);

        $this->assertEquals($estimatorA->medoids(), $estimatorB->medoids());
    }
}
