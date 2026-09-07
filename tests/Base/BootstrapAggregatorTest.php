<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Base;

use Generator;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RunInSeparateProcess;
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
use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Swoole;
use Rubix\ML\Specifications\ExtensionIsLoaded;

#[Group('MetaEstimators')]
#[CoversClass(BootstrapAggregator::class)]
class BootstrapAggregatorTest extends TestCase
{
    protected const int TRAIN_SIZE = 512;

    protected const int TEST_SIZE = 256;

    protected const float MIN_SCORE = 0.9;

    protected const int RANDOM_SEED = 0;

    protected SwissRoll $generator;

    protected BootstrapAggregator $estimator;

    protected RSquared $metric;

    protected ?Backend $backend = null;

    /**
     * @return Generator<string, array{backend: Backend}>
     */
    public static function provideBackends() : Generator
    {
        $serialBackend = new Serial();

        yield (string) $serialBackend => [
            'backend' => $serialBackend,
        ];

        $ampBackend = new Amp();

        yield (string) $ampBackend => [
            'backend' => $ampBackend,
        ];

        if (ExtensionIsLoaded::with('swoole')->passes()) {
            $swooleBackend = new Swoole();

            yield (string) $swooleBackend => [
                'backend' => $swooleBackend,
            ];
        }
    }

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

    protected function tearDown() : void
    {
        $this->backend?->shutdown();
    }

    #[Test]
    public function preConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    #[Test]
    public function type() : void
    {
        $this->assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    #[Test]
    public function compatibility() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    public function params() : void
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
    #[Test]
    #[RunInSeparateProcess]
    public function trainPredict(Backend $backend) : void
    {
        $this->backend = $backend;

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

    #[Test]
    #[TestDox('Predictions are identical regardless of the backend')]
    public function predictIsBackendAgnostic() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->setBackend(new Serial());

        $this->estimator->train($training);

        $expected = $this->estimator->predict($testing);

        $amp = new Amp();
        $this->backend = $amp;

        $this->estimator->setBackend($amp);

        $this->assertSame($expected, $this->estimator->predict($testing));
    }

    #[Test]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }

    #[Test]
    #[TestDox('Backend is transient and resolved lazily')]
    public function backendIsTransient() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->setBackend(new Serial());

        $this->estimator->train($training);

        self::assertTrue($this->estimator->trained());

        self::assertArrayNotHasKey('backend', $this->estimator->__serialize());

        $copy = unserialize(serialize($this->estimator));

        self::assertInstanceOf(BootstrapAggregator::class, $copy);
        self::assertTrue($copy->trained());

        $predictions = $copy->predict($training);

        self::assertCount(self::TRAIN_SIZE, $predictions);

        self::assertArrayNotHasKey('backend', $copy->__serialize());
    }
}
