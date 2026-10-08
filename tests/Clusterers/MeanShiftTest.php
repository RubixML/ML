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
use Rubix\ML\Clusterers\MeanShift;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Clusterers\Seeders\Random;
use Rubix\ML\Clusterers\Seeders\Preset;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Clusterers')]
#[CoversClass(MeanShift::class)]
class MeanShiftTest extends TestCase
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

    protected MeanShift $estimator;

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

        $this->estimator = new MeanShift(
            radius: 66,
            ratio: 0.1,
            epochs: 100,
            minShift: 1e-4,
            tree: new BallTree(),
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
    public function progressContract() : void
    {
        $this->assertSame([], iterator_to_array($this->estimator->progress(), false));

        srand(self::RANDOM_SEED);

        $estimator = new MeanShift(
            radius: 66,
            ratio: 0.1,
            epochs: 5,
            minShift: 1e-6,
            tree: new BallTree(),
            seeder: new Random()
        );

        $estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $estimator->train($training);

        $this->assertTrue($estimator->trained());

        $losses = $estimator->losses();

        $this->assertIsArray($losses);
        $this->assertNotEmpty($losses);

        $rows = iterator_to_array($estimator->progress(), false);

        $this->assertCount(count($losses), $rows);

        foreach ($rows as $row) {
            $this->assertIsInt($row['Epoch']);
            $this->assertIsFloat($row['Shift']);
        }

        $this->assertSame(array_values($losses), array_column($rows, 'Shift'));
        $this->assertSame(array_keys($losses), array_column($rows, 'Epoch'));
    }

    #[Test]
    public function badRadius() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new MeanShift(radius: 0.0);
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
            'radius' => 66.0,
            'ratio' => 0.1,
            'epochs' => 100,
            'min shift' => 1e-4,
            'tree' => new BallTree(),
            'seeder' => new Random(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    public function estimateRadius() : void
    {
        $subset = $this->generator->generate(intdiv(self::TRAIN_SIZE, 4));

        $radius = MeanShift::estimateRadius(dataset: $subset);

        $this->assertIsFloat($radius);
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
        $this->assertContainsOnlyArray($centroids);

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
    public function trainWithOutlyingPresetCentroids() : void
    {
        $presets = [];

        for ($i = 0; $i < 20; ++$i) {
            $presets[] = [100 + $i * 100, 100 + $i * 100];
        }

        $estimator = new MeanShift(1.0, 1.0, 10, 1e-4, new BallTree(), new Preset($presets));

        $training = Unlabeled::quick([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
        ]);

        $estimator->train($training);

        $this->assertTrue($estimator->trained());
        $this->assertSame($presets, $estimator->centroids());
    }

    /**
     * The shift reported for an epoch is the total displacement of the centroid
     * candidates from the exact positions they were shifted from.
     *
     * The candidates in this fixture are seeded more than the radius apart and
     * each one only ever sees the pair of samples in its own neighborhood, so no
     * candidate is ever pruned and consecutive epochs can be compared directly.
     */
    #[Test]
    public function lossesMatchCentroidDisplacement() : void
    {
        $presets = [];
        $samples = [];

        for ($i = 0; $i < 20; ++$i) {
            $presets[] = [(float) ($i * 20)];

            $samples[] = [(float) ($i * 20 + 3)];
            $samples[] = [(float) ($i * 20 + 4)];
        }

        $training = Unlabeled::quick($samples);

        foreach ([3.0, 5.0, 7.0] as $radius) {
            foreach ([2, 3, 4, 5, 6] as $epochs) {
                $previous = new MeanShift(
                    radius: $radius,
                    ratio: 0.5,
                    epochs: $epochs - 1,
                    minShift: 0.0,
                    tree: new BallTree(),
                    seeder: new Preset($presets)
                );

                $previous->train($training);

                $estimator = new MeanShift(
                    radius: $radius,
                    ratio: 0.5,
                    epochs: $epochs,
                    minShift: 0.0,
                    tree: new BallTree(),
                    seeder: new Preset($presets)
                );

                $estimator->train($training);

                $this->assertSame(
                    count($estimator->centroids()),
                    count($previous->centroids()),
                    'Fixture pruned candidates, radius '
                    . "$radius, epochs $epochs"
                );

                $expected = $this->displacement(
                    current: $estimator->centroids(),
                    previous: $previous->centroids()
                ) / $training->numSamples();

                $this->assertEqualsWithDelta(
                    $expected,
                    $estimator->losses()[$epochs],
                    1e-12,
                    "Radius $radius epoch $epochs"
                );
            }
        }
    }

    /**
     * Candidates that were pruned by the merge step of an epoch are not charged
     * for the shift of the candidate that absorbed them, and the candidates that
     * survived are charged from the positions they were shifted from rather than
     * from their former index in the unmerged list.
     *
     * The candidates in this fixture are 5 apart so 9, 13, and 16 of them are
     * pruned during the first two epochs at radii 6, 14, and 22 respectively.
     */
    #[Test]
    public function lossesExcludePrunedCandidates() : void
    {
        $samples = [];

        for ($i = 0; $i < 20; ++$i) {
            $samples[] = [(float) ($i * 5.0)];
        }

        $samples[5] = [26.0];
        $samples[6] = [34.0];

        $training = Unlabeled::quick($samples);

        $expected = [
            [6.0, [
                1 => 0.368520038173,
                2 => 0.033673603439,
                3 => 0.006231837823,
                4 => 0.001252842273,
                5 => 0.000294386435,
                6 => 0.000086803173,
            ]],
            [14.0, [
                1 => 0.469491981338,
                2 => 0.125865079631,
                3 => 0.113513533954,
                4 => 0.026387599012,
                5 => 0.006151814003,
                6 => 0.001435147581,
            ]],
            [22.0, [
                1 => 0.461785166053,
                2 => 0.155143397263,
                3 => 0.129400532072,
                4 => 0.122586570570,
                5 => 0.033391727535,
                6 => 0.009132488142,
            ]],
        ];

        foreach ($expected as [$radius, $losses]) {
            $estimator = new MeanShift(
                radius: $radius,
                ratio: 0.05,
                epochs: 6,
                minShift: 0.0,
                tree: new BallTree(),
                seeder: new Preset($samples)
            );

            $estimator->train($training);

            $this->assertLessThan(count($samples), count($estimator->centroids()));

            $this->assertSame(array_keys($losses), array_keys($estimator->losses()));

            foreach ($losses as $epoch => $loss) {
                $this->assertEqualsWithDelta(
                    $loss,
                    $estimator->losses()[$epoch],
                    1e-9,
                    "Radius $radius epoch $epoch"
                );
            }
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

        $this->estimator->predict(Unlabeled::quick());
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

    /**
     * Compute the total per-column displacement between two sets of centroids.
     *
     * @param list<(int|float)[]> $current
     * @param list<(int|float)[]> $previous
     * @return float
     */
    protected function displacement(array $current, array $previous) : float
    {
        $displacement = 0.0;

        foreach ($current as $cluster => $centroid) {
            foreach ($centroid as $column => $mean) {
                $displacement += abs($previous[$cluster][$column] - $mean);
            }
        }

        return $displacement;
    }
}
