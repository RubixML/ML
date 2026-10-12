<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Clusterers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Clusterers\OPTICS;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

use function Rubix\ML\argmax;
use function array_column;
use function array_map;
use function array_sum;
use function is_infinite;
use function range;
use function sort;

#[Group('Clusterers')]
#[CoversClass(OPTICS::class)]
class OPTICSTest extends TestCase
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

    protected OPTICS $estimator;

    protected VMeasure $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            [
                'inner' => new Circle(x: 0.0, y: 0.0, scale: 1.0, noise: 0.01),
                'middle' => new Circle(x: 0.0, y: 0.0, scale: 5.0, noise: 0.05),
                'outer' => new Circle(x: 0.0, y: 0.0, scale: 10.0, noise: 0.1),
            ]
        );

        $this->estimator = new OPTICS(
            radius: 3.0,
            minDensity: 10,
            steepness: 0.3,
            minClusterSize: 20,
            weighted: false,
            tree: new BallTree()
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
    public function badRadius() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new OPTICS(radius: 0.0);
    }

    #[Test]
    public function badMinDensity() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new OPTICS(minDensity: 0);
    }

    #[Test]
    public function badSteepness() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new OPTICS(steepness: 1.0);
    }

    #[Test]
    public function badMinClusterSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new OPTICS(minClusterSize: 0);
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
            'radius' => 3.0,
            'min density' => 10,
            'steepness' => 0.3,
            'min cluster size' => 20,
            'weighted' => false,
            'tree' => new BallTree(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    public function trainPredict() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $testing->labels()
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function orderingAndReachability() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $ordering = $this->estimator->ordering();

        $this->assertCount(self::TRAIN_SIZE, $ordering);
        $this->assertContainsOnlyInt($ordering);

        $sorted = $ordering;
        sort($sorted);

        $this->assertSame(range(0, self::TRAIN_SIZE - 1), $sorted);

        $reachability = $this->estimator->reachability();

        $this->assertCount(self::TRAIN_SIZE, $reachability);
        $this->assertContainsOnlyFloat($reachability);
        $this->assertTrue(is_infinite($reachability[$ordering[0]]));

        $coreDistances = $this->estimator->coreDistances();

        $this->assertCount(self::TRAIN_SIZE, $coreDistances);
        $this->assertContainsOnlyFloat($coreDistances);
    }

    #[Test]
    public function progressContract() : void
    {
        $this->assertSame([], iterator_to_array($this->estimator->progress(), false));

        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $rows = iterator_to_array($this->estimator->progress(), false);

        $this->assertCount(self::TRAIN_SIZE, $rows);

        foreach ($rows as $row) {
            $this->assertIsInt($row['Point']);
            $this->assertIsFloat($row['Reachability']);
        }

        $ordering = $this->estimator->ordering();
        $reachability = $this->estimator->reachability();

        $this->assertSame($ordering, array_column($rows, 'Point'));

        $expected = array_map(
            fn (int $point) => $reachability[$point],
            $ordering
        );

        $this->assertSame($expected, array_column($rows, 'Reachability'));
    }

    #[Test]
    public function predictsNoiseForIsolatedSamples() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $this->assertSame(
            [OPTICS::NOISE],
            $this->estimator->predict(Unlabeled::quick(samples: [[0.0, 1000.0]]))
        );
    }

    #[Test]
    public function proba() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        foreach ([false, true] as $weighted) {
            $estimator = new OPTICS(
                radius: 3.0,
                minDensity: 10,
                steepness: 0.3,
                minClusterSize: 20,
                weighted: $weighted,
                tree: new BallTree()
            );

            $estimator->train($training);

            $proba = $estimator->proba($testing);

            $predictions = $estimator->predict($testing);

            $this->assertCount($testing->numSamples(), $proba);

            foreach ($proba as $i => $dist) {
                $this->assertIsArray($dist);
                $this->assertContainsOnlyFloat($dist);
                $this->assertEqualsWithDelta(1.0, array_sum($dist), 1e-8);
                $this->assertSame($predictions[$i], argmax($dist));
            }
        }
    }

    #[Test]
    public function probaForIsolatedSamples() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        [$dist] = $this->estimator->proba(Unlabeled::quick(samples: [[0.0, 1000.0]]));

        $this->assertSame(1.0, $dist[OPTICS::NOISE]);

        foreach ($dist as $cluster => $probability) {
            if ($cluster !== OPTICS::NOISE) {
                $this->assertSame(0.0, $probability);
            }
        }
    }

    #[Test]
    public function probaUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->proba(Unlabeled::quick(samples: [[1.0, 2.0]]));
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

        $this->estimator->predict(Unlabeled::quick(samples: [[1.0, 2.0]]));
    }

    #[Test]
    #[TestDox('Throws an exception when predicting with incompatible data')]
    public function predictIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

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
}
