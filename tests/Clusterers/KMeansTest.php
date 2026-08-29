<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Clusterers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Clusterers\Seeders\PlusPlus;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

use function array_sum;
use function min;

#[Group('Clusterers')]
#[CoversClass(KMeans::class)]
class KMeansTest extends TestCase
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
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected Agglomerate $generator;

    protected KMeans $estimator;

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

        $this->estimator = new KMeans(
            k:3,
            batchSize: 128,
            epochs: 300,
            minChange: 1e-4,
            kernel: new Euclidean(),
            seeder: new PlusPlus()
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

        new KMeans(k: 0);
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
            'batch size' => 128,
            'epochs' => 300,
            'min change' => 1e-4,
            'kernel' => new Euclidean(),
            'seeder' => new PlusPlus(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    public function trainPartialPredict() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $folds = $training->stratifiedFold(3);

        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

        $this->assertTrue($this->estimator->trained());

        $centroids = $this->estimator->centroids();

        $this->assertIsArray($centroids);
        $this->assertCount(3, $centroids);
        $this->assertContainsOnlyArray($centroids);

        $sizes = $this->estimator->sizes();

        $this->assertIsArray($sizes);
        $this->assertCount(3, $sizes);
        $this->assertContainsOnlyInt($sizes);

        $total = $folds[0]->numSamples() + $folds[1]->numSamples() + $folds[2]->numSamples();

        $this->assertSame($total, array_sum($sizes));
        $this->assertGreaterThanOrEqual(0, min($sizes));

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
    public function onlineLearning() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $sizes = $this->estimator->sizes();

        $this->assertSame(self::TRAIN_SIZE, array_sum($sizes));
        $this->assertGreaterThanOrEqual(0, min($sizes));

        $batch = $this->generator->generate(100);

        $this->estimator->partial($batch);

        $sizes = $this->estimator->sizes();

        $this->assertCount(3, $sizes);
        $this->assertContainsOnlyInt($sizes);
        $this->assertSame(self::TRAIN_SIZE + $batch->numSamples(), array_sum($sizes));
        $this->assertGreaterThanOrEqual(0, min($sizes));

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function onlineLearningKeepsSizesValid() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $total = self::TRAIN_SIZE;

        for ($round = 0; $round < 3; ++$round) {
            $batch = $this->generator->generate(50);

            $total += $batch->numSamples();

            $this->estimator->partial($batch);

            $sizes = $this->estimator->sizes();

            $this->assertCount(3, $sizes);
            $this->assertContainsOnlyInt($sizes);
            $this->assertSame($total, array_sum($sizes));
            $this->assertGreaterThanOrEqual(0, min($sizes));
        }
    }

    #[Test]
    public function partialWithoutTrain() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->partial($training);

        $this->assertTrue($this->estimator->trained());

        $sizes = $this->estimator->sizes();

        $this->assertCount(3, $sizes);
        $this->assertContainsOnlyInt($sizes);
        $this->assertSame(self::TRAIN_SIZE, array_sum($sizes));
        $this->assertGreaterThanOrEqual(0, min($sizes));
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
}
