<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Classifiers;

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
use Rubix\ML\Classifiers\OneVsRest;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Swoole;
use Rubix\ML\Specifications\ExtensionIsLoaded;

#[Group('Classifiers')]
#[CoversClass(OneVsRest::class)]
class OneVsRestTest extends TestCase
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

    protected OneVsRest $estimator;

    protected FBeta $metric;

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

        $ampBackend = new Amp(4);

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

        $this->estimator = new OneVsRest(new GaussianNB());

        $this->metric = new FBeta();

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
            'base' => new GaussianNB(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[DataProvider('provideBackends')]
    #[Test]
    #[RunInSeparateProcess]
    public function trainPredictProba(Backend $backend) : void
    {
        $this->backend = $backend;

        $this->estimator->setBackend($backend);

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

        self::assertInstanceOf(OneVsRest::class, $copy);
        self::assertTrue($copy->trained());

        $predictions = $copy->predict($training);

        self::assertCount(self::TRAIN_SIZE, $predictions);

        self::assertArrayNotHasKey('backend', $copy->__serialize());
    }
}
