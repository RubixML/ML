<?php

declare(strict_types=1);

namespace Rubix\ML\Tests;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\BootstrapAggregator;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Datasets\Generators\SwissRoll;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Tests\DataProvider\BackendProviderTrait;

#[Group('MetaEstimators')]
#[CoversClass(BootstrapAggregator::class)]
class BootstrapAggregatorTest extends TestCase
{
    use BackendProviderTrait;

    protected const int TRAIN_SIZE = 512;

    protected const int TEST_SIZE = 256;

    protected const float MIN_SCORE = 0.9;

    protected const int RANDOM_SEED = 0;

    protected SwissRoll $generator;

    protected BootstrapAggregator $estimator;

    protected RSquared $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new SwissRoll(x: 4.0, y: -7.0, z: 0.0, scale: 1.0, depth: 0.3);

        $this->estimator = new BootstrapAggregator(
            new RegressionTree(maxHeight: 10),
            estimators: 30,
            ratio: 0.5
        );

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    public function testAssertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    public function testType() : void
    {
        $this->assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    public function testCompatibility() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    public function testParams() : void
    {
        $expected = [
            'base' => new RegressionTree(maxHeight: 10),
            'estimators' => 30,
            'ratio' => 0.5,
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    /**
     * @param Backend $backend
     */
    #[DataProvider('provideBackends')]
    public function trainPredict(Backend $backend) : void
    {
        $this->estimator->setBackend($backend);

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        /** @var list<int|float> $labels */
        $labels = $testing->labels();
        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function testPredictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}
