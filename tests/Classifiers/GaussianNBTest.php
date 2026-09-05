<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Classifiers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Classifiers')]
#[CoversClass(GaussianNB::class)]
class GaussianNBTest extends TestCase
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

    protected GaussianNB $estimator;

    protected FBeta $metric;

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

        $this->estimator = new GaussianNB(priors: null, smoothing: 1e-8);

        $this->metric = new FBeta();

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
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
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
            'priors' => null,
            'smoothing' => 1e-8,
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    public function trainPartialPredict() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $folds = $training->stratifiedFold(3);

        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

        $this->assertTrue($this->estimator->trained());

        $priors = $this->estimator->priors();

        $this->assertIsArray($priors);
        $this->assertCount(3, $priors);
        $this->assertContainsOnlyFloat($priors);

        $means = $this->estimator->means();

        $this->assertIsArray($means);
        $this->assertCount(3, $means);
        $this->assertContainsOnlyArray($means);

        $variances = $this->estimator->variances();

        $this->assertIsArray($variances);
        $this->assertCount(3, $variances);
        $this->assertContainsOnlyArray($variances);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $testing->labels()
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function partialWithAbsentClassesPreservesSmoothing() : void
    {
        $estimator = new GaussianNB(smoothing: 1e-6);

        $training = (new Labeled(
            samples: (new Blob(center: [10.0, 10.0], stdDev: 4.0))->generate(64)->samples(),
            labels: array_fill(0, 64, 'red')
        ))->merge(new Labeled(
            samples: (new Blob(center: [40.0, 40.0], stdDev: 8.0))->generate(64)->samples(),
            labels: array_fill(0, 64, 'green')
        ));

        $estimator->train($training);

        $baseline = [
            'means' => $estimator->means(),
            'variances' => $estimator->variances(),
        ];

        $batch = new Labeled(
            samples: (new Blob(center: [80.0, 10.0], stdDev: 6.0))->generate(32)->samples(),
            labels: array_fill(0, 32, 'blue')
        );

        $estimator->partial($batch);

        $firstUpdate = $estimator->variances();

        for ($i = 0; $i < 5; ++$i) {
            $estimator->partial($batch);
        }

        $this->assertEquals($baseline['means']['red'], $estimator->means()['red']);
        $this->assertEquals($baseline['means']['green'], $estimator->means()['green']);
        $this->assertEquals($baseline['variances']['red'], $estimator->variances()['red']);
        $this->assertEquals($baseline['variances']['green'], $estimator->variances()['green']);

        $this->assertEqualsWithDelta(
            $firstUpdate['blue'],
            $estimator->variances()['blue'],
            1e-9
        );
    }

    #[Test]
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']], labels: ['green']));
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
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $restored = unserialize(serialize($this->estimator));

        $this->assertTrue($restored->trained());
        $this->assertEquals($this->estimator->priors(), $restored->priors());
        $this->assertEquals($this->estimator->means(), $restored->means());
        $this->assertEquals($this->estimator->predict($testing), $restored->predict($testing));
    }

    #[Test]
    public function probaRowsSumToOne() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $probabilities = $this->estimator->proba($testing);

        $this->assertCount(self::TEST_SIZE, $probabilities);

        foreach ($probabilities as $probability) {
            $this->assertEqualsWithDelta(1.0, array_sum($probability), 1e-8);
        }
    }
}
