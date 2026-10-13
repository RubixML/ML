<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Clusterers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Clusterers\Ward;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

use function count;
use function array_sum;
use function iterator_to_array;

use const Rubix\ML\EPSILON;

#[Group('Clusterers')]
#[CoversClass(Ward::class)]
class WardTest extends TestCase
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

    protected Ward $estimator;

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

        $this->estimator = new Ward(k: 3);

        $this->metric = new VMeasure();

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function preConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    #[Test]
    public function progressContract() : void
    {
        $this->assertSame([], iterator_to_array($this->estimator->progress(), false));

        srand(self::RANDOM_SEED);

        $estimator = new Ward(k: 3);

        $estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $estimator->train($training);

        $this->assertTrue($estimator->trained());

        $losses = $estimator->losses();

        $this->assertIsArray($losses);
        $this->assertNotEmpty($losses);
        $this->assertCount($training->numSamples() - 3, $losses);

        $rows = iterator_to_array($estimator->progress(), false);

        $this->assertCount(count($losses), $rows);

        foreach ($rows as $row) {
            $this->assertIsInt($row['Epoch']);
            $this->assertIsFloat($row['Loss']);
        }
    }

    #[Test]
    public function badK() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Ward(k: 0);
    }

    #[Test]
    public function badMaxDistance() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Ward(k: 3, maxDistance: -0.5);
    }

    #[Test]
    public function tooFewSamples() : void
    {
        $estimator = new Ward(k: 5);

        $this->expectException(InvalidArgumentException::class);

        $estimator->train(Unlabeled::quick(samples: [[1.0], [2.0], [3.0], [4.0]]));
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
            'maxDistance' => INF,
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

        $centroids = $this->estimator->centroids();

        $this->assertIsArray($centroids);
        $this->assertCount(3, $centroids);
        $this->assertContainsOnlyArray($centroids);

        $sizes = $this->estimator->sizes();

        $this->assertIsArray($sizes);
        $this->assertCount(3, $sizes);
        $this->assertEquals($training->numSamples(), array_sum($sizes));

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnlyFloat($losses);
        $this->assertCount($training->numSamples() - 3, $losses);

        $predictions = $this->estimator->predict($testing);

        $this->assertCount(self::TEST_SIZE, $predictions);

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $testing->labels()
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function trainKEqualsNumSamples() : void
    {
        srand(self::RANDOM_SEED);

        $training = Unlabeled::quick(samples: [[1.0], [2.0], [3.0], [4.0], [5.0]]);

        $estimator = new Ward(k: 5);

        $estimator->train($training);

        $this->assertTrue($estimator->trained());

        $this->assertCount(5, $estimator->centroids());
        $this->assertSame([], $estimator->losses());

        $sizes = $estimator->sizes();

        foreach ($sizes as $size) {
            $this->assertSame(1, $size);
        }

        $predictions = $estimator->predict($training);

        $this->assertCount(5, $predictions);

        $this->assertEqualsCanonicalizing(range(0, 4), $predictions);
    }

    #[Test]
    public function trainStopsEarlyAtMaxDistance() : void
    {
        $training = Unlabeled::quick(samples: [[0.0], [1.0], [2.0], [5.0]]);

        $estimator = new Ward(k: 1, maxDistance: 1.5);

        $estimator->train($training);

        $this->assertTrue($estimator->trained());

        $this->assertCount(2, $estimator->centroids());

        $losses = $estimator->losses();

        $this->assertIsArray($losses);
        $this->assertCount(2, $losses);

        foreach ($losses as $cost) {
            $this->assertLessThanOrEqual(1.5, $cost);
        }
    }

    #[Test]
    public function trainNoMergeWithTinyMaxDistance() : void
    {
        $training = Unlabeled::quick(samples: [[1.0], [2.0], [3.0]]);

        $estimator = new Ward(k: 1, maxDistance: EPSILON);

        $estimator->train($training);

        $this->assertTrue($estimator->trained());

        $this->assertCount(3, $estimator->centroids());
        $this->assertSame([], $estimator->losses());

        $sizes = $estimator->sizes();

        foreach ($sizes as $size) {
            $this->assertSame(1, $size);
        }
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
    public function deterministic() : void
    {
        srand(self::RANDOM_SEED);
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $estimatorA = new Ward(k: 3);
        $estimatorA->train($training);

        $estimatorB = new Ward(k: 3);
        $estimatorB->train($training);

        $this->assertEquals($estimatorA->centroids(), $estimatorB->centroids());
    }

    #[Test]
    public function proba() : void
    {
        srand(self::RANDOM_SEED);

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $testing = $this->generator->generate(self::TEST_SIZE);

        $proba = $this->estimator->proba($testing);

        $this->assertCount(self::TEST_SIZE, $proba);

        foreach ($proba as $distributions) {
            $this->assertCount(3, $distributions);

            foreach ($distributions as $p) {
                $this->assertGreaterThanOrEqual(0.0, $p);
                $this->assertLessThanOrEqual(1.0, $p);
            }
        }
    }
}
